#!/bin/bash
#SBATCH --job-name=expJ
#SBATCH --partition=gpua10_hi
#SBATCH --output=expJ_%j.out
#SBATCH --error=expJ_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1

# Exp J -- una corrida del estudio de carbonos totales en login-1 / A10.
# Entrena Y evalua el mismo config en UN solo job: menos sbatch, y elimina la
# clase de error "evalue un checkpoint que todavia no habia terminado".
#
# Uso:
#   sbatch run_train_j.sh config_j_a.yaml     # experimental (con degeneracion)
#   sbatch run_train_j.sh config_j_0.yaml     # control (sin degeneracion)
#
# Las dos de una:
#   for c in config_j_a.yaml config_j_0.yaml; do sbatch run_train_j.sh "$c"; done
#
# ANTES de lanzar: subir por scp al base_dir del cluster los dos archivos que
# se generan localmente con prep/ --
#   vectors_19v_totales_202465.npy
#   peaks_pkl_deg_202465.npz

source /home/lpassaglia.iquir/anaconda3/etc/profile.d/conda.sh
conda activate /home/lpassaglia.iquir/anaconda3/envs/NMR_env

# Ajustar esta ruta a donde hayas clonado el repo en el cluster.
cd ~/nmr-hsqc-to-vector-/experiments/J_carbonos_totales

CONFIG="${1:?Falta el config, ej: sbatch run_train_j.sh config_j_a.yaml}"

echo "=== EXP J | CONFIG: $CONFIG ==="

echo "=== FASE 1/2: TRAIN ==="
python -u train.py --config "$CONFIG"
if [ $? -ne 0 ]; then
    echo "[ABORT] train.py fallo -- no se evalua (evitar reportar la EMA de un checkpoint viejo)"
    exit 1
fi

echo "=== FASE 2/2: EVAL ==="
# --oraculo all: cruda + asistida v1 + asistida v2, tabla de 3 vias.
python -u evaluate.py --config "$CONFIG" --oraculo all --batch-size 256
