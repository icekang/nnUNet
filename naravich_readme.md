# Datasets
All the data and results of nnUNet can be found in [Naravich's engaging folder](https://engaging-ood.mit.edu/pun/sys/files/fs/nfs/erelab001/shared/Computational_Group/Naravich/nnUNet_Datasets/)
- Lumen and wall OCT
    - [Dataset300_Lumen_and_Wall_OCT.py](nnunetv2/dataset_conversion/Dataset300_Lumen_and_Wall_OCT.py)
- Calcium OCT 
    - [Dataset302_Calcium_OCT.py](nnunetv2/dataset_conversion/Dataset302_Calcium_OCTv2.py)
## Dataset Setup
- Download the dataset from the link above
- Set environment variables to where you save it. For example, in `~/.bashrc`:
    - `nnUNet_raw`="/home/`<user>`/datasets/nnUNet_Datasets/nnUNet_raw"
    - `nnUNet_preprocessed`="/home/`<user>`/datasets/nnUNet_Datasets/nnUNet_preprocessed"
    - `nnUNet_results`="/home/`<user>`/datasets/nnUNet_Datasets/nnUNet_results"
    - For more information, see [set_environment_variables.md](documentation/set_environment_variables.md)
- You are all set! Now you can run nnUNet commands with the dataset.

# Running nnUNet
- Users are recommended to checkout [the nnUNet documentation](readme.md) especially,
    - [Installation instructions](documentation/installation_instructions.md) for how to properly install nnUNet.
    - [Dataset conversion](documentation/dataset_format.md) for how the above datasets are converted to nnUNet format.
    - [Usage instructions](documentation/how_to_use_nnunet.md) for how to train, predict, and evaluate models.
    - Additionally, all the script submitted to the cluster can be found in [slurm_script_archive](slurm_scripts_archives).

## Lumen and wall Supervised Pre-training
- Checkout nnUNet's [supervised pre-training documentation](documentation/pretraining_and_finetuning.md) for more information.
- Submitted script can be found in 
    - LaW OCT pre-training
        - [2D nnUNet](slurm_scripts_archives/preprocess_LaW_OCT_submission_2d.sh)
        - [3D nnUNet](slurm_scripts_archives/preprocess_LaW_OCT_submission_3d.sh)
    - Fine-tune LaW OCT on Calcium OCT
        - 2D nnUNet
            - [Fold 0 (finetune_LaW_to_Calcium_OCT_submission_2d)](slurm_scripts_archives/finetune_LaW_to_Calcium_OCT_submission_2d.sh)
            - [Fold 1 script (finetune_LaW_to_Calcium_OCT_submission_2d_fold1.sh)](slurm_scripts_archives/finetune_LaW_to_Calcium_OCT_submission_2d_fold1.sh)
            - [Fold 2 script (finetune_LaW_to_Calcium_OCT_submission_2d_fold2.sh)](slurm_scripts_archives/finetune_LaW_to_Calcium_OCT_submission_2d_fold2.sh)
        - 3D nnUNet (32x160x128)
            - [Fold 0 (train_Calcium_OCT_submission_3d_32x160x128_b10_fold0.sh)](slurm_scripts_archives/train_Calcium_OCT_submission_3d_32x160x128_b10_fold0.sh)
            - [Fold 1 (train_Calcium_OCT_submission_3d_32x160x128_b10_fold0.sh)](slurm_scripts_archives/train_Calcium_OCT_submission_3d_32x160x128_b10_fold1.sh)
            - [Fold 2 (train_Calcium_OCT_submission_3d_32x160x128_b10_fold0.sh)](slurm_scripts_archives/train_Calcium_OCT_submission_3d_32x160x128_b10_fold2.sh)

# What's Scale Analysis?

Scale analysis is an analysis on scale of dataset and its effect on the model performance. For calcium OCT, there are 3 scales: 33% (1 train, 1 val), 66% (3 train, 1 val), 100% (4 train, 2 val)
- `splits_final_2.json` can be found in `nnUNet_preprocessed`/Dataset302_Calcium_OCTv2/splits_final_2.json
- `splits_final_4.json` can be found in `nnUNet_preprocessed`/Dataset302_Calcium_OCTv2/splits_final_4.json
- where `nnUNet_preprocessed` is the environment variable

The trainers for scale analysis are defined in the following:
- [33% nnUNetTrainerScaleAnalysis2.py](nnunetv2/training/nnUNetTrainer/nnUNetTrainerScaleAnalysis2.py)
- [66% nnUNetTrainerScaleAnalysis4.py](nnunetv2/training/nnUNetTrainer/nnUNetTrainerScaleAnalysis4.py)
- [100% nnUNetTrainer.py](nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py) which is the default trainer

An example of a submitted script for scale analysis can be found in the following
- [scale_analysis_2_Calcium_OCT__LaW_and_then_from_scratch_3d_32x160x128_b10_fold0.sh](slurm_scripts_archives/scale_analysis_2_Calcium_OCT__LaW_and_then_from_scratch_3d_32x160x128_b10_fold0.sh) which basically just specify `-tr` (TRAINER) to the desired trainer

# FAQ
- Weight and Bias is not working?
    - If you are using SLURM, it is not connected to the internet. Therefore, you need to run `wandb offline` and manually sync them later.
- Log of my SLURM script freezes at ``unpacking dataset..``
    - Check the actaul result in `nnUNet_results` folder. It is likely that the script is still running but the log is not updating.
