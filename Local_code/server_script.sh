#!/bin/bash

#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -o slurm-test.out
#SBATCH -J my_job

python classification_city_data.py