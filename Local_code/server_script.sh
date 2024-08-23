#!/bin/bash

#SBATCH -c 4
#SBATCH --gres=gpu:2
#SBATCH -o slurm-test.out
#SBATCH -J my_job

python cnn_modle_classification.py