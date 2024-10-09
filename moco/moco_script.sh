#!/bin/bash

#SBATCH -c 2
#SBATCH --gres=gpu:2
#SBATCH -o train_moco.out
#SBATCH -J train_embeddings

python extract_baseset.py