DATASET_ID=302
for fold in $(seq 0 4); do
	nnUNetv2_train -tr GenesisTrainer $DATASET_ID 3d_fullres $fold --npz
done
