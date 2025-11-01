#!/bin/bash

for FOLD in 0 1 2 3 4; do
  for SUFFIX in 20 40 60 80; do
    SCRIPT="train_Dataset310_3d_fold${FOLD}_SA${SUFFIX}.sh"
    if [ -f "$SCRIPT" ]; then
      echo "Submitting $SCRIPT"
      sbatch "$SCRIPT"
    else
      echo "File $SCRIPT not found!"
    fi
  done
done