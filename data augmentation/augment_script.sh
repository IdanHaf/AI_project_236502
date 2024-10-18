#!/bin/bash

#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o augment.out
#SBATCH -J augment_preprocess

python augment.py