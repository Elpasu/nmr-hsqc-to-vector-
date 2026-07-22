#!/bin/bash
#SBATCH --job-name=expE3_scaling
#SBATCH --partition=gpua10_hi
#SBATCH --output=expE3_scaling_%j.out
#SBATCH --error=expE3_scaling_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1

source /home/lpassaglia.iquir/anaconda3/etc/profile.d/conda.sh
conda activate /home/lpassaglia.iquir/anaconda3/envs/NMR_env

# Ajustar esta ruta a donde hayas clonado el repo en el cluster.
cd ~/nmr-hsqc-to-vector-/experiments/E3_dos_conjuntos

# Pasar el config de la fraccion a entrenar como argumento:
#   sbatch run_train_scaling.sh config_scaling_10.yaml
#   sbatch run_train_scaling.sh config_scaling_25.yaml
#   sbatch run_train_scaling.sh config_scaling_50.yaml
#   sbatch run_train_scaling.sh config_scaling_75.yaml
#   sbatch run_train_scaling.sh config_scaling_100.yaml
CONFIG="${1:?Falta el config, ej: sbatch run_train_scaling.sh config_scaling_10.yaml}"
python -u train.py --config "$CONFIG"
