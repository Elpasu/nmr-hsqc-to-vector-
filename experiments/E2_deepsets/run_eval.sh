#!/bin/bash
#SBATCH --job-name=expE2_eval
#SBATCH --partition=gpua10_hi
#SBATCH --output=expE2_eval_%j.out
#SBATCH --error=expE2_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

source /home/lpassaglia.iquir/anaconda3/etc/profile.d/conda.sh
conda activate /home/lpassaglia.iquir/anaconda3/envs/NMR_env

# Ajustar esta ruta a donde hayas clonado el repo en el cluster.
cd ~/nmr-hsqc-to-vector-/experiments/E2_deepsets

python -u evaluate.py --config config.yaml --oraculo both --batch-size 256
