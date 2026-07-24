#!/bin/bash
#SBATCH --job-name=expE3_eval_xpu
#SBATCH --partition=gpunode
#SBATCH --output=expE3_eval_xpu_%j.out
#SBATCH --error=expE3_eval_xpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:intel_xt1550:1
#
# Evaluacion de E3 en Clementina XXI (Intel XPU).
# run_eval.sh sigue siendo el de login-1 (A10/CUDA), intacto.
# Ver docs/MIGRACION_XPU_Clementina_XXI.md
#
#   sbatch run_eval_clementina.sh                            # settransformer
#   sbatch run_eval_clementina.sh config_settransformer.yaml

set -euo pipefail

# Ver seccion 10.1 del documento: sin este unset, SLURM oculta los tiles.
unset ZE_AFFINITY_MASK || true
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

# Ver la nota sobre CONDA_SH en run_train_settransformer_clementina.sh.
CONDA_SH="${CONDA_SH:-/data/contrib/pci_78/envs/miniconda3/etc/profile.d/conda.sh}"
if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: no encuentro conda.sh en $CONDA_SH" >&2
    echo "       Corri 'conda info --base' en el login y exporta:" >&2
    echo "       export CONDA_SH=<base>/etc/profile.d/conda.sh" >&2
    exit 1
fi
source "$CONDA_SH"
conda activate /data/contrib/pci_78/envs/nmr_xpu

export NMR_DATA_DIR=/data/contrib/pci_78/Lucas/DB_202K
export NMR_DEVICE=xpu

cd /home/lpassaglia/nmr-hsqc-to-vector-/experiments/E3_dos_conjuntos

CONFIG="${1:-config_settransformer.yaml}"

# --oraculo all: cruda + asistida v1 (doble) + asistida v2 (hetero), tabla 3-vias.
python -u evaluate.py --config "$CONFIG" --oraculo all --batch-size 256
