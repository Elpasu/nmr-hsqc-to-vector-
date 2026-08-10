#!/bin/bash
#SBATCH --job-name=expE3_hp_xpu
#SBATCH --partition=gpunode
#SBATCH --output=expE3_hp_xpu_%j.out
#SBATCH --error=expE3_hp_xpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=12:30:00
#SBATCH --gres=gpu:intel_xt1550:1
#
# Exp I -- una corrida del estudio de hiperparametros en Clementina XXI (Intel
# XPU). Mismo config, mismo train.py/evaluate.py que run_sweep.sh (login-1/A10)
# -- run_sweep.sh sigue siendo el de login-1, intacto. Entrena Y evalua el
# mismo config en UN solo job (misma razon que run_sweep.sh: menos sbatch, y
# elimina la clase de error "evalue un checkpoint que todavia no termino").
# Ver docs/MIGRACION_XPU_Clementina_XXI.md
#
# IMPORTANTE: las 23 corridas del sweep tienen que quedar en UN SOLO cluster
# (login-1 O Clementina, no mezclados) -- la banda de ruido de las 3 replicas
# de seed mide variacion de semilla; si se mezcla hardware, esa banda queda
# confundida con la variacion A10-vs-XPU (~0.8pp, ya documentada en RESULTS.md,
# seccion de migracion XPU) y deja de servir para decidir que es senal.
#
# Uso:
#   sbatch run_sweep_clementina.sh hp_sweep/configs/hp_dmodel_128.yaml
#
# Los 23 de una:
#   for cfg in hp_sweep/configs/*.yaml; do sbatch run_sweep_clementina.sh "$cfg"; done

set -euo pipefail

# Ver seccion 10.1 del documento de migracion: sin este unset, SLURM setea mal
# ZE_AFFINITY_MASK y oculta los tiles -- PyTorch puede no ver la GPU.
unset ZE_AFFINITY_MASK || true
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# Ver la nota sobre CONDA_SH en run_train_settransformer_clementina.sh.
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: no encuentro conda.sh en $CONDA_SH" >&2
    echo "       Corri 'conda info --base' en el login y exporta:" >&2
    echo "       export CONDA_SH=<base>/etc/profile.d/conda.sh" >&2
    exit 1
fi
# Los hooks internos de conda/oneAPI (ej. mpivars.deactivate.sh) no son
# compatibles con `set -u` -- ver run_train_settransformer_clementina.sh para
# el detalle del error real que esto evita (job 1489555).
set +u
source "$CONDA_SH"
conda activate /data/contrib/pci_78/envs/nmr_xpu
set -u

export NMR_DATA_DIR="${NMR_DATA_DIR:-/data/contrib/pci_78/Lucas/DB_202K}"
export NMR_DEVICE=xpu     # exige XPU: si no la ve, aborta en vez de usar CPU

NMR_REPO="${NMR_REPO:-$HOME/nmr-hsqc-to-vector-}"
cd "$NMR_REPO/experiments/E3_dos_conjuntos"

CONFIG="${1:?Falta el config, ej: sbatch run_sweep_clementina.sh hp_sweep/configs/hp_dmodel_128.yaml}"

echo "=== EXP I (XPU) | CONFIG: $CONFIG ==="

echo "=== FASE 1/2: TRAIN ==="
python -u train.py --config "$CONFIG"

echo "=== FASE 2/2: EVAL ==="
# --oraculo all: cruda + asistida v1 + asistida v2, con la tabla de 3 vias que
# collect_results.py parsea. `set -e` ya aborta el job si train.py fallo, asi
# que si llegamos hasta aca el checkpoint es real.
python -u evaluate.py --config "$CONFIG" --oraculo all --batch-size 256
