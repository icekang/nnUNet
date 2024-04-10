DATASET_ID=302

# for fold in $(seq 0 2); do
# 	nnUNetv2_train -tr GenesisNNUNetTrainer $DATASET_ID 3d_fullres $fold --val_best
# done

for fold in $(seq 0 2); do
	nnUNetv2_train -tr nnUNetTrainerScaleAnalysis2 $DATASET_ID 3d_fullres $fold --val_best
done

for fold in $(seq 0 2); do
	nnUNetv2_train -tr nnUNetTrainerScaleAnalysis2 $DATASET_ID 2d $fold --val_best
done

for fold in $(seq 0 2); do
	nnUNetv2_train -tr nnUNetTrainerScaleAnalysis4 $DATASET_ID 3d_fullres $fold --val_best
done

for fold in $(seq 0 2); do
	nnUNetv2_train -tr nnUNetTrainerScaleAnalysis4 $DATASET_ID 2d $fold --val_best
done
