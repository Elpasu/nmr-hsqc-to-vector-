#!/bin/bash
#SBATCH --job-name=expE3_hp
#SBATCH --partition=gpua10_hi
#SBATCH --output=expE3_hp_%j.out
#SBATCH --error=expE3_hp_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1

# Exp I -- una corrida del estudio de hiperparametros: entrena Y evalua el
# mismo config en UN solo job. Que sean uno solo (y no dos) es deliberado:
# 23 sbatch en vez de 46, y elimina la clase de error "evalue un checkpoint
# que todavia no habia terminado de entrenarse".
#
# Uso:
#   sbatch run_sweep.sh hp_sweep/configs/hp_dmodel_128.yaml
#
# Los 23 de una:
#   for cfg in hp_sweep/configs/*.yaml; do sbatch run_sweep.sh "$cfg"; done

source /home/lpassaglia.iquir/anaconda3/etc/profile.d/conda.sh
conda activate /home/lpassaglia.iquir/anaconda3/envs/NMR_env

# Ajustar esta ruta a donde hayas clonado el repo en el cluster.
cd ~/nmr-hsqc-to-vector-/experiments/E3_dos_conjuntos

CONFIG="${1:?Falta el config, ej: sbatch run_sweep.sh hp_sweep/configs/hp_dmodel_128.yaml}"

echo "=== EXP I | CONFIG: $CONFIG ==="

echo "=== FASE 1/2: TRAIN ==="
python -u train.py --config "$CONFIG"
if [ $? -ne 0 ]; then
    echo "[ABORT] train.py fallo -- no se evalua (evitar reportar la EMA de un checkpoint viejo)"
    exit 1
fi

echo "=== FASE 2/2: EVAL ==="
# --oraculo all: cruda + asistida v1 + asistida v2, con la tabla de 3 vias que
# collect_results.py parsea.
python -u evaluate.py --config "$CONFIG" --oraculo all --batch-size 256
