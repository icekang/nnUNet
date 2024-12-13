#!/bin/bash

# Activating the conda environment
source activate nnunet
which python
export OPENBLAS_NUM_THREADS=1

# Setup env variables nn_UNet
export nnUNet_raw="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_raw"
export nnUNet_preprocessed="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_preprocessed"
export nnUNet_results="/home/gridsan/nchutisilp/datasets/nnUNet_Datasets/nnUNet_results"

CONFIG=3d_32x160x128_b10
DATASET_ID=302
TRAINER=nnUNetTrainer

INPUT_DIR=$nnUNet_raw/Dataset306_Sohee_Ajay_Calcium_OCT/imagesTr
LABEL_DIR=$nnUNet_raw/Dataset306_Sohee_Ajay_Calcium_OCT/labelsTr

rename_result_folder_to_active_folder() {
    mv ${nnUNet_results}/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}_no_pretrained ${nnUNet_results}/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}
}

rename_active_folder_to_result_folder() {
    mv ${nnUNet_results}/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD} ${nnUNet_results}/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}_no_pretrained
}

FOLD=0
rename_result_folder_to_active_folder
OUTPUT_DIR=$nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}_on_Sohee_Ajay_Calcium_OCT
nnUNetv2_predict -i $INPUT_DIR -o $OUTPUT_DIR -f $FOLD -c $CONFIG -device cuda -d $DATASET_ID -tr $TRAINER -chk checkpoint_best.pth
nnUNetv2_evaluate_simple $LABEL_DIR $OUTPUT_DIR -l 1
rename_active_folder_to_result_folder

FOLD=1
rename_result_folder_to_active_folder
OUTPUT_DIR=$nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}_on_Sohee_Ajay_Calcium_OCT
nnUNetv2_predict -i $INPUT_DIR -o $OUTPUT_DIR -f $FOLD -c $CONFIG -device cuda -d $DATASET_ID -tr $TRAINER -chk checkpoint_best.pth
nnUNetv2_evaluate_simple $LABEL_DIR $OUTPUT_DIR -l 1
mv $nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD} $nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}_no_pretrained
rename_active_folder_to_result_folder

FOLD=2
rename_result_folder_to_active_folder
OUTPUT_DIR=$nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}_on_Sohee_Ajay_Calcium_OCT
nnUNetv2_predict -i $INPUT_DIR -o $OUTPUT_DIR -f $FOLD -c $CONFIG -device cuda -d $DATASET_ID -tr $TRAINER -chk checkpoint_best.pth
nnUNetv2_evaluate_simple $LABEL_DIR $OUTPUT_DIR -l 1
mv $nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD} $nnUNet_results/Dataset302_Calcium_OCTv2/${TRAINER}__nnUNetPlans__${CONFIG}/fold_${FOLD}_no_pretrained
rename_active_folder_to_result_folder
