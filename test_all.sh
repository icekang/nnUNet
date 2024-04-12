DATASET_ID=302

# for fold in $(seq 0 2); do
#     nnUNetv2_predict -i /storage_bizon/naravich/nnUNet_Datasets/nnUNet_raw/Dataset302_Calcium_OCTv2/imagesTs -o /storage_bizon/naravich/nnUNet_Datasets/nnUNet_results/Dataset302_Calcium_OCTv2/nnUNetTrainer__nnUNetPlans__2d/fold_${fold}_test -f $fold -c 2d -device cuda -d $DATASET_ID
# done

# for fold in $(seq 0 2); do
#     echo "FOLD $fold"
#     nnUNetv2_evaluate_simple /storage_bizon/naravich/nnUNet_Datasets/nnUNet_raw/Dataset302_Calcium_OCTv2/labelsTs/ /storage_bizon/naravich/nnUNet_Datasets/nnUNet_results/Dataset302_Calcium_OCTv2/nnUNetTrainer__nnUNetPlans__2d/fold_${fold}_test/ -l 1
#     echo "FOLD $fold Completed"
# done

DATASET_ID=302
for fold in $(seq 0 2); do
    for CONFIG in 3d_fullres 2d; do
        for TRAINER in nnUNetTrainer; do
            echo "FOLD $fold CONFIG $CONFIG TRAINER $TRAINER"
            INPUT_DIR=/storage_bizon/naravich/nnUNet_Datasets/nnUNet_raw/Dataset302_Calcium_OCTv2/imagesTs
            OUTPUT_DIR=/storage_bizon/naravich/nnUNet_Datasets/nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${fold}_test
            LABEL_DIR=/storage_bizon/naravich/nnUNet_Datasets/nnUNet_raw/Dataset302_Calcium_OCTv2/labelsTs

            nnUNetv2_predict -i $INPUT_DIR -o $OUTPUT_DIR -f $fold -c $CONFIG -device cuda -d $DATASET_ID -tr $TRAINER
            nnUNetv2_evaluate_simple $LABEL_DIR $OUTPUT_DIR -l 1
            echo "Completed FOLD $fold CONFIG $CONFIG TRAINER $TRAINER"
        done
    done
done
# for fold in $(seq 0 2); do
#     for CONFIG in 3d_fullres 2d; do
#         for TRAINER in nnUNetTrainerScaleAnalysis2 nnUNetTrainerScaleAnalysis4; do
#             echo "FOLD $fold CONFIG $CONFIG TRAINER $TRAINER"
#             INPUT_DIR=/storage_bizon/naravich/nnUNet_Datasets/nnUNet_raw/Dataset302_Calcium_OCTv2/imagesTs
#             OUTPUT_DIR=/storage_bizon/naravich/nnUNet_Datasets/nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${fold}_test
#             LABEL_DIR=/storage_bizon/naravich/nnUNet_Datasets/nnUNet_raw/Dataset302_Calcium_OCTv2/labelsTs

#             nnUNetv2_predict -i $INPUT_DIR -o $OUTPUT_DIR -f $fold -c $CONFIG -device cuda -d $DATASET_ID -tr $TRAINER
#             nnUNetv2_evaluate_simple $LABEL_DIR $OUTPUT_DIR -l 1
#             echo "Completed FOLD $fold CONFIG $CONFIG TRAINER $TRAINER"
#         done
#     done
# done

