#!/bin/bash

DATASET_ID=300

for fold in $(seq 0 2); do
    for CONFIG in 2d 3d_fullres; do
        nnUNetv2_train -tr nnUNetTrainerEarlyStopping $DATASET_ID 2d $fold --val_best --c
        echo "Sleeping for 2 minutes"
        sleep 2m
    done
done
