DATASET_ID=300

# export nnUNet_def_n_proc=10
export nnUNet_n_proc_DA=0

for fold in $(seq 0 2); do
	nnUNetv2_train -tr nnUNetTrainerScaleAnalysis4 $DATASET_ID 2d $fold --val_best --c
    echo "Sleeping for 2 minutes"
    sleep 2m
done
