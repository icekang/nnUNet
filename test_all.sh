
DATASET_ID=300
for fold in $(seq 0 2); do
    for CONFIG in 3d_fullres; do
        for TRAINER in nnUNetTrainerScaleAnalysis2 nnUNetTrainerScaleAnalysis4 nnUNetTrainerEarlyStopping; do
            echo "FOLD $fold CONFIG $CONFIG TRAINER $TRAINER"
            INPUT_DIR=/storage_bizon/naravich/nnUNet_Datasets/nnUNet_raw/Dataset300_Lumen_and_Wall_OCT/imagesTs
            OUTPUT_DIR=/storage_bizon/naravich/nnUNet_Datasets/nnUNet_results/Dataset300_Lumen_and_Wall_OCT/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${fold}_test
            LABEL_DIR=/storage_bizon/naravich/nnUNet_Datasets/nnUNet_raw/Dataset300_Lumen_and_Wall_OCT/labelsTs

            nnUNetv2_predict -i $INPUT_DIR -o $OUTPUT_DIR -f $fold -c $CONFIG -device cuda -d $DATASET_ID -tr $TRAINER -chk checkpoint_best.pth
            nnUNetv2_evaluate_simple $LABEL_DIR $OUTPUT_DIR -l 1 2
            echo "Completed FOLD $fold CONFIG $CONFIG TRAINER $TRAINER"
        done
    done
done
