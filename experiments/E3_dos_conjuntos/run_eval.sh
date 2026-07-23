#!/bin/bash
#SBATCH --job-name=expE3_eval
#SBATCH --partition=gpua10_hi
#SBATCH --output=expE3_eval_%j.out
#SBATCH --error=expE3_eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

source /home/lpassaglia.iquir/anaconda3/etc/profile.d/conda.sh
conda activate /home/lpassaglia.iquir/anaconda3/envs/NMR_env

# Ajustar esta ruta a donde hayas clonado el repo en el cluster.
cd ~/nmr-hsqc-to-vector-/experiments/E3_dos_conjuntos

# Pasar el config a evaluar como argumento (default: deepsets).
#   sbatch run_eval.sh config_deepsets.yaml
#   sbatch run_eval.sh config_settransformer.yaml
CONFIG="${1:-config_deepsets.yaml}"
# --oraculo all: cruda + asistida v1 (doble) + asistida v2 (hetero) con tabla 3-vias.
python -u evaluate.py --config "$CONFIG" --oraculo all --batch-size 256
