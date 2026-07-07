import os
import sys
import json
import torch
import numpy as np
import argparse
from pathlib import Path
import SimpleITK as sitk

# Set environment variables for nnUNet programmatically
os.environ['nnUNet_raw'] = "/nfs/erelab001/shared/Computational_Group/Naravich/datasets/nnUNet_Datasets/nnUNet_raw"
os.environ['nnUNet_preprocessed'] = "/nfs/erelab001/shared/Computational_Group/Naravich/datasets/nnUNet_Datasets/nnUNet_preprocessed"
os.environ['nnUNet_results'] = "/nfs/erelab001/shared/Computational_Group/Naravich/datasets/nnUNet_Datasets/nnUNet_results"

# Add the project root directory to the python path to load libraries correctly
PROJECT_ROOT = Path('/nfs/erelab001/shared/Computational_Group/Naravich/nnUNet')
sys.path.append(str(PROJECT_ROOT))

# Import nnUNet utilities
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from analysis.data_utils import enhance_reduce_texture, adjust_intensity

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Dice coefficient for label 1 (calcium)."""
    # Flatten and convert to boolean mask
    mask_true = (y_true == 1)
    mask_pred = (y_pred == 1)
    intersection = np.sum(mask_true & mask_pred)
    total = np.sum(mask_true) + np.sum(mask_pred)
    if total == 0:
        return 1.0
    return 2.0 * intersection / total

def write_float_image(image_np: np.ndarray, output_fname: str, properties: dict) -> None:
    """Save a float32 image to NIfTI format preserving geometry metadata."""
    if image_np.ndim == 4:
        image_np = image_np[0]
    itk_image = sitk.GetImageFromArray(image_np.astype(np.float32, copy=False))
    itk_image.SetSpacing(properties['sitk_stuff']['spacing'])
    itk_image.SetOrigin(properties['sitk_stuff']['origin'])
    itk_image.SetDirection(properties['sitk_stuff']['direction'])
    sitk.WriteImage(itk_image, output_fname, True)

def perturb_texture(image_np: np.ndarray, factor: float) -> np.ndarray:
    """Apply texture perturbation (enhancement or reduction) to the image array."""
    if factor == 1.0:
        return image_np.copy()
    if factor < 1.0:
        mode = 'reduce'
        actual_factor = 1.0 / factor
    else:
        mode = 'enhance'
        actual_factor = factor
    return enhance_reduce_texture(image_np, enhancement_factor=actual_factor, mode=mode, method='unsharp', adaptive=False)

def get_model_config(model_name: str, fold: int) -> dict:
    if model_name == 'CLIP':
        return {
            'results_dir': Path("/nfs/erelab001/shared/Computational_Group/Naravich/datasets/nnUNet_Datasets/nnUNet_results/Dataset307_Sohee_Calcium_OCT_CrossValidation/nnUNetTrainer__nnUNetPlans__3d_32x160x128_b10"),
            'fold_subdir': f"fold_{fold}_CLIP_PrePostStent_Pretrained"
        }
    elif model_name == 'Genesis':
        fold_subdir = f"fold_{fold}_Genesis_Pretrained"
        if fold in (1, 2, 3, 4):
            fold_subdir = f"{fold_subdir}/fold_{fold}"
        return {
            'results_dir': Path("/nfs/erelab001/shared/Computational_Group/Naravich/datasets/nnUNet_Datasets/nnUNet_results/Dataset307_Sohee_Calcium_OCT_CrossValidation/nnUNetTrainer__nnUNetPlans__3d_32x160x128_b10"),
            'fold_subdir': fold_subdir
        }
    elif model_name == 'LaW':
        return {
            'results_dir': Path("/nfs/erelab001/shared/Computational_Group/Naravich/datasets/nnUNet_Datasets/nnUNet_results/Dataset309_Sohee_Calcium_OCT_CrossValidation/nnUNetTrainer__nnUNetPlans__3d_32x160x128_b10"),
            'fold_subdir': f"fold_{fold}_LaW_Pretrained"
        }
    elif model_name == 'No_Pretrain':
        return {
            'results_dir': Path("/nfs/erelab001/shared/Computational_Group/Naravich/datasets/nnUNet_Datasets/nnUNet_results/Dataset308_Sohee_Calcium_OCT_CrossValidation/nnUNetTrainer__nnUNetPlans__3d_32x160x128_b10"),
            'fold_subdir': f"fold_{fold}_No_Pretrained"
        }
    raise ValueError(f"Unknown model name {model_name} or invalid fold {fold}")

def main():
    parser = argparse.ArgumentParser(description="Run perturbation evaluation for specific folds")
    parser.add_argument('--fold', type=str, default='0', help="Fold to evaluate (0, 1, 2, 3, 4, or 'all')")
    args = parser.parse_args()

    # Setup directories
    output_dir = Path("/home/naravich/nnunet_texture_ablations")
    perturbed_images_dir = output_dir / 'perturbed_images'
    perturbed_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset configurations
    raw_images_dir = Path("/nfs/erelab001/shared/Computational_Group/Naravich/datasets/nnUNet_Datasets/nnUNet_raw/Dataset307_Sohee_Calcium_OCT_CrossValidation/imagesTr")
    gt_dir = Path("/nfs/erelab001/shared/Computational_Group/Naravich/datasets/nnUNet_Datasets/nnUNet_preprocessed/Dataset307_Sohee_Calcium_OCT_CrossValidation/gt_segmentations")
    splits_file = Path("/nfs/erelab001/shared/Computational_Group/Naravich/datasets/nnUNet_Datasets/nnUNet_preprocessed/Dataset307_Sohee_Calcium_OCT_CrossValidation/splits_final.json")
    
    # Load splits file to get validation cases dynamically
    with open(splits_file, 'r') as f:
        splits = json.load(f)
        
    if args.fold.lower() == 'all':
        folds_to_run = [0, 1, 2, 3, 4]
    else:
        try:
            folds_to_run = [int(args.fold)]
        except ValueError:
            print(f"Invalid fold argument: {args.fold}. Must be an integer 0-4 or 'all'.")
            sys.exit(1)
            
    # Gather validation cases for requested folds
    validation_cases = []
    for fold in folds_to_run:
        validation_cases.extend(splits[fold]['val'])
    
    # Define factors
    texture_factors = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    intensity_factors = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    print("--------------------------------------------------")
    print(f"1. Generating and saving perturbed images for folds: {folds_to_run}...")
    print("--------------------------------------------------")
    
    # Pre-generate and save all perturbed images (shared across all models)
    perturbed_files = {}
    for case in validation_cases:
        perturbed_files[case] = {
            'texture': {},
            'intensity': {},
            'gt': gt_dir / f"{case}.nii.gz",
            'original': raw_images_dir / f"{case}_0000.nii.gz"
        }
        
        # Load original image
        img, props = SimpleITKIO().read_images([str(perturbed_files[case]['original'])])
        
        # Save texture perturbations
        for factor in texture_factors:
            out_name = perturbed_images_dir / f"{case}_texture_{factor}.nii.gz"
            perturbed_files[case]['texture'][factor] = out_name
            if not out_name.exists():
                print(f"Generating texture variant for {case} at factor {factor}...")
                perturbed_img = perturb_texture(img, factor)
                write_float_image(perturbed_img, str(out_name), props)
            
        # Save intensity perturbations
        for factor in intensity_factors:
            out_name = perturbed_images_dir / f"{case}_intensity_{factor}.nii.gz"
            perturbed_files[case]['intensity'][factor] = out_name
            if not out_name.exists():
                print(f"Generating intensity variant for {case} at factor {factor}...")
                perturbed_img = adjust_intensity(img, factor)
                write_float_image(perturbed_img, str(out_name), props)

    print("Perturbed images generation completed!")
    
    print("\n--------------------------------------------------")
    print("2. Running model evaluations...")
    print("--------------------------------------------------")
    
    for model_name in ['No_Pretrain', 'CLIP', 'Genesis', 'LaW']:
        print(f"\nEvaluating Model: {model_name}...")
        
        for fold in folds_to_run:
            print(f"\nEvaluating Fold {fold}...")
            
            # Setup output directories for this fold
            fold_out_dir = output_dir / model_name / f"fold_{fold}"
            dice_scores_dir = fold_out_dir / "dice_scores"
            predictions_dir = fold_out_dir / "predictions"
            
            try:
                config = get_model_config(model_name, fold)
            except ValueError as e:
                print(f"Warning: {e}. Skipping fold {fold} for model {model_name}!")
                continue
                
            # Path to checkpoints
            results_dir = config['results_dir']
            fold_subdir = config['fold_subdir']
            
            # Check if the fold folder exists
            fold_dir_path = results_dir / fold_subdir
            if not fold_dir_path.exists():
                print(f"Warning: Checkpoint path {fold_dir_path} does not exist. Skipping fold {fold} for model {model_name}!")
                continue
                
            # Create temporary symlink 'fold_{fold}' pointing to 'fold_subdir' inside results_dir
            need_symlink = (fold_subdir != f'fold_{fold}')
            symlink_path = results_dir / f'fold_{fold}'
            if need_symlink:
                if symlink_path.exists() or symlink_path.is_symlink():
                    symlink_path.unlink()
                symlink_path.symlink_to(results_dir / fold_subdir, target_is_directory=True)
            
            try:
                # Initialize predictor
                print(f"Loading nnUNetPredictor for {model_name} (Fold {fold})...")
                predictor = nnUNetPredictor(
                    tile_step_size=0.5,
                    use_gaussian=True,
                    use_mirroring=False,  # Set to False to speed up inference
                    perform_everything_on_device=True,
                    device=DEVICE,
                    verbose=False,
                    verbose_preprocessing=False,
                    allow_tqdm=True
                )
                
                predictor.initialize_from_trained_model_folder(
                    str(results_dir),
                    use_folds=(fold,),
                    checkpoint_name='checkpoint_best.pth'
                )
                
                # Create directories only if model is found and predictor initialized successfully
                dice_scores_dir.mkdir(parents=True, exist_ok=True)
                predictions_dir.mkdir(parents=True, exist_ok=True)
                
                # Predict for each case of this fold
                fold_cases = splits[fold]['val']
                for case in fold_cases:
                    dice_out_file = dice_scores_dir / f"{case}.json"
                    pred_out_file = predictions_dir / f"{case}.pth"
                    
                    # Check if results are already computed
                    if dice_out_file.exists() and pred_out_file.exists():
                        print(f"Results for {model_name} fold {fold} case {case} already exist. Skipping prediction.")
                        continue
                    
                    print(f"Running prediction for {case}...")
                    
                    # Load ground truth segmentation
                    gt_image, _ = SimpleITKIO().read_images([str(perturbed_files[case]['gt'])])
                    
                    # Texture predictions
                    texture_dice_scores = []
                    texture_preds = []
                    for factor in texture_factors:
                        perturbed_path = perturbed_files[case]['texture'][factor]
                        img, props = SimpleITKIO().read_images([str(perturbed_path)])
                        
                        # Run inference
                        pred = predictor.predict_single_npy_array(img, props, None, None, False)
                        
                        # Calculate Dice
                        dice = dice_coefficient(gt_image, pred)
                        texture_dice_scores.append(dice)
                        texture_preds.append(pred.astype(np.uint8))
                        print(f"  Texture Factor {factor}: Dice = {dice:.4f}")
                        
                    # Intensity predictions
                    intensity_dice_scores = []
                    intensity_preds = []
                    for factor in intensity_factors:
                        perturbed_path = perturbed_files[case]['intensity'][factor]
                        img, props = SimpleITKIO().read_images([str(perturbed_path)])
                        
                        # Run inference
                        pred = predictor.predict_single_npy_array(img, props, None, None, False)
                        
                        # Calculate Dice
                        dice = dice_coefficient(gt_image, pred)
                        intensity_dice_scores.append(dice)
                        intensity_preds.append(pred.astype(np.uint8))
                        print(f"  Intensity Factor {factor}: Dice = {dice:.4f}")
                    
                    # Save dice scores to JSON (extremely fast to load)
                    print(f"Saving dice scores to {dice_out_file}...")
                    dice_data = {
                        'case_id': case,
                        'texture_factors': texture_factors,
                        'texture_dice_scores': texture_dice_scores,
                        'intensity_factors': intensity_factors,
                        'intensity_dice_scores': intensity_dice_scores
                    }
                    with open(dice_out_file, 'w') as f_json:
                        json.dump(dice_data, f_json, indent=4)
                    
                    # Save predictions and gt to pth (heavy arrays)
                    print(f"Saving predictions to {pred_out_file}...")
                    torch.save({
                        'case_id': case,
                        'texture_predictions': texture_preds,
                        'intensity_predictions': intensity_preds,
                        'gt': gt_image.astype(np.uint8)
                    }, pred_out_file)
                    
                # Cleanup predictor and empty GPU cache
                del predictor
                torch.cuda.empty_cache()
                
            finally:
                # Clean up the symlink
                if need_symlink:
                    if symlink_path.is_symlink() or symlink_path.exists():
                        symlink_path.unlink()
                
    print("\n--------------------------------------------------")
    print("Evaluation completed successfully!")
    print("--------------------------------------------------")

if __name__ == '__main__':
    main()
