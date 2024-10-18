#!/bin/bash

#SBATCH -c 2
#SBATCH --gres=gpu:2
#SBATCH -o train_moco.out
#SBATCH -J train_embeddings

python compare_nn.py