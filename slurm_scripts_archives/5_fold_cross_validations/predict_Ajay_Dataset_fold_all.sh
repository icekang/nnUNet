#!/bin/bash

source activate nnunet
which python

export OPENBLAS_NUM_THREADS=1

CONFIG=3d_32x160x128_b10
DATASET_ID=302
DATASET_NAME=Calcium_OCTv2
TRAINER=nnUNetTrainer
PLANS=nnUNetPlans
FOLD=all

PREDICTION_INPUT_DIR=/home/gridsan/nchutisilp/datasets/Shreya_Calcium_Hand_Segmentation_Check/Coreg_Patient_Raw_Flattened
PREDICTION_OUTPUT_DIR=/home/gridsan/nchutisilp/datasets/Shreya_Calcium_Hand_Segmentation_Check/Coreg_Patient_Raw_Flattened_Calcium_Prediction

PRETRAIN_NAME=pretrained_with_LaW
CHECKPOINT_DIRECTORY_PATH=${nnUNet_results}/Dataset${DATASET_ID}_${DATASET_NAME}/${TRAINER}__${PLANS}__${CONFIG}/fold_${FOLD}

# mv ${CHECKPOINT_DIRECTORY_PATH}_${PRETRAIN_NAME} ${CHECKPOINT_DIRECTORY_PATH}

# nnUNetv2_predict -i $PREDICTION_INPUT_DIR -o $PREDICTION_OUTPUT_DIR -d $DATASET_ID -c $CONFIG -f $FOLD -p $PLANS -tr $TRAINER -device cuda -chk checkpoint_best.pth

# mv ${CHECKPOINT_DIRECTORY_PATH} ${CHECKPOINT_DIRECTORY_PATH}_${PRETRAIN_NAME}

LABEL_DIR=/home/gridsan/nchutisilp/datasets/Shreya_Calcium_Hand_Segmentation_Check/Coreg_Seg_Corrected_Calcium_Labels_Only

# Put predictions that exist in the label directory to ${PREDICTION_OUTPUT_DIR}/HasLabels
# mkdir -p ${PREDICTION_OUTPUT_DIR}/HasLabels
# for file in ${PREDICTION_OUTPUT_DIR}/All/*.nii.gz; do
#     base_name=$(basename $file)
#     if [ -f ${LABEL_DIR}/$base_name ]; then
#         # echo $base_name
#         cp $file ${PREDICTION_OUTPUT_DIR}/HasLabels/
#     fi
# done

nnUNetv2_evaluate_simple $LABEL_DIR ${PREDICTION_OUTPUT_DIR}/HasLabels -l 1