#!/bin/bash
#SBATCH --job-name=expE2_train
#SBATCH --partition=gpua10_hi
#SBATCH --output=expE2_train_%j.out
#SBATCH --error=expE2_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:1

source /home/lpassaglia.iquir/anaconda3/etc/profile.d/conda.sh
conda activate /home/lpassaglia.iquir/anaconda3/envs/NMR_env

# Ajustar esta ruta a donde hayas clonado el repo en el cluster.
cd ~/nmr-hsqc-to-vector-/experiments/E2_deepsets

python -u train.py --config config.yaml
