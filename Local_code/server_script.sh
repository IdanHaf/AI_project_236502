#!/bin/bash

#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -o slurm-test.out
#SBATCH -J tune_job

python language_classifier.py