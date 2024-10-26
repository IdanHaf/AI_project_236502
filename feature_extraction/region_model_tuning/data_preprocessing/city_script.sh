#!/bin/bash

#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o cities.out
#SBATCH -J cities_preprocess

python dataset_preprocessing.py