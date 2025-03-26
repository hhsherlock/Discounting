#! /bin/bash

source /home/yaning/Documents/python_env/pyro/bin/activate
export CUDA_VISIBLE_DEVICES=0,1 
python3 multiply_MCMC_hierarchical_parallel_0.py
