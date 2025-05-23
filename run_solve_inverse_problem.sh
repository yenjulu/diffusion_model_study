#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=23:00:00

python run_solve_inverse_problem.py 