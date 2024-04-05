DATASET_ID=301

for fold in $(seq 0 4); do
    nnUNetv2_predict -i /storage_bizon/naravich/nnUNet_Datasets/nnUNet_raw/Dataset301_Calcium_OCT/imagesTs -o /storage_bizon/naravich/nnUNet_Datasets/nnUNet_results/Dataset301_Calcium_OCT/nnUNetTrainer__nnUNetPlans__2d/fold_${fold}_test -f $fold -c 2d -device cuda -d 301
done

for fold in $(seq 0 4); do
    echo "FOLD $fold"
    nnUNetv2_evaluate_simple /storage_bizon/naravich/nnUNet_Datasets/nnUNet_raw/Dataset301_Calcium_OCT/labelsTs/ /storage_bizon/naravich/nnUNet_Datasets/nnUNet_results/Dataset301_Calcium_OCT/nnUNetTrainer__nnUNetPlans__2d/fold_${fold}_test/ -l 1
    echo "FOLD $fold Completed"
done

