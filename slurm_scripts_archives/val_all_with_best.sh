
DATASET_ID=301

for fold in $(seq 2 4); do
	nnUNetv2_train $DATASET_ID 3d_fullres $fold --val --val_best
    sleep 1m
done

for fold in $(seq 0 4); do
	nnUNetv2_train $DATASET_ID 3d_lowres $fold --val --val_best
    sleep 1m
done

for fold in $(seq 0 4); do
	nnUNetv2_train $DATASET_ID 2d $fold --val --val_best
    sleep 1m
done

for fold in $(seq 0 4); do
	nnUNetv2_train $DATASET_ID 3d_cascade_fullres $fold --val --val_best
    sleep 1m
done
