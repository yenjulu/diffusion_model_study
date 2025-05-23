#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=23:00:00

python main.py \
  --config=configs/ve/dwi_200_ncsnpp_continuous.py \
  --eval_folder=eval/20_dir \
  --mode='train' \
  --workdir=workdir/20_dir   # DWI_9d3s, shepp_logan_patches_4c