#!/bin/bash

#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -o test.out
#SBATCH -J test

python run_testset.py