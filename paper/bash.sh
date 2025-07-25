#!/bin/bash

source /etc/profile.d/conda.sh
conda activate spike

# Optional: set GPU device (e.g., use only GPU 0)
export CUDA_VISIBLE_DEVICES=0

# Run the Python script
python /home/yaning/Documents/Discounting/paper/parameter_recovery_tanh.py