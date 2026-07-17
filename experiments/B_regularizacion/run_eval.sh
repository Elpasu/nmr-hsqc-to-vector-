#!/bin/bash
#SBATCH --job-name=expB_eval
#SBATCH --partition=gpua10_hi
#SBATCH --output=expB_eval_%j.out
#SBATCH --error=expB_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

source /home/lpassaglia.iquir/anaconda3/etc/profile.d/conda.sh
conda activate /home/lpassaglia.iquir/anaconda3/envs/NMR_env

# Ajustar esta ruta a donde hayas clonado el repo en el cluster.
cd ~/nmr-hsqc-to-vector-/experiments/B_regularizacion

python -u evaluate.py --config config.yaml --oraculo both --batch-size 256
