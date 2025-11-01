#!/bin/bash

for FOLD in 0 1 2 3 4; do
  # Submit full trainer
  SCRIPT="fold${FOLD}_full_Genesis.sh"
  if [ -f "$SCRIPT" ]; then
    echo "Submitting $SCRIPT"
    sbatch "$SCRIPT"
  else
    echo "File $SCRIPT not found!"
  fi

  # Submit scale analysis trainers
  for SUFFIX in SA20 SA40 SA60 SA80; do
    SCRIPT="fold${FOLD}_${SUFFIX}_Genesis.sh"
    if [ -f "$SCRIPT" ]; then
      echo "Submitting $SCRIPT"
      sbatch "$SCRIPT"
    else
      echo "File $SCRIPT not found!"
    fi
  done
done
