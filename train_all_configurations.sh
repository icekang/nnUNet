DATASET_ID=300
for fold in `seq 0 4`
do
nnUNetv2_train $DATASET_ID 3d_lowres $fold --npz;
done
