DATASET_ID=301

# for fold in $(seq 0 4); do
# 	nnUNetv2_train $DATASET_ID 3d_fullres $fold --npz
# done

# for fold in $(seq 1 4); do
# 	nnUNetv2_train $DATASET_ID 3d_lowres $fold --npz
# done
#nnUNetv2_train $DATASET_ID 2d 2 --npz --c

#for fold in $(seq 3 4); do
#	nnUNetv2_train $DATASET_ID 2d $fold --npz
#done

#for fold in $(seq 0 4); do
#	nnUNetv2_train $DATASET_ID 3d_cascade_fullres $fold --npz
#done

#nnUNetv2_train $DATASET_ID 3d_cascade_fullres 4
#sleep 15m

#nnUNetv2_train $DATASET_ID 2d 0 --npz --c

#sleep 15m
nnUNetv2_train -tr GenesisTrainer $DATASET_ID 3d_fullres 3 --npz --c --val_best
for fold in $(seq 4 4); do
	nnUNetv2_train -tr GenesisTrainer $DATASET_ID 3d_fullres $fold --npz --val_best
	# sleep 15m
done
