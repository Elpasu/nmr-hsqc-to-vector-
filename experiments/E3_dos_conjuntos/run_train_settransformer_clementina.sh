#!/bin/bash
#SBATCH --job-name=expE3_st_train_xpu
#SBATCH --partition=gpunode
#SBATCH --output=expE3_st_train_xpu_%j.out
#SBATCH --error=expE3_st_train_xpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:intel_xt1550:1
#
# Entrenamiento de E3 Set Transformer en Clementina XXI (Intel XPU).
# run_train_settransformer.sh sigue siendo el de login-1 (A10/CUDA), intacto.
# Ver docs/MIGRACION_XPU_Clementina_XXI.md
#
#   sbatch run_train_settransformer_clementina.sh

set -euo pipefail

# --- Level Zero -----------------------------------------------------------
# SLURM setea mal ZE_AFFINITY_MASK en Clementina y oculta los tiles: sin este
# unset, PyTorch puede no ver la GPU. Ver seccion 10.1 del documento.
unset ZE_AFFINITY_MASK || true
export ZE_FLAT_DEVICE_HIERARCHY=FLAT   # 8 tiles de 64 GB; E3 usa uno solo

# --- Entorno --------------------------------------------------------------
# En un job no-interactivo conda NO esta inicializado: hay que sourcear su
# profile.d/conda.sh antes de `conda activate`. Cada usuario tiene su propio
# conda en el HOME (verificado: `conda info --base` -> /home/<user>/miniconda3),
# por eso se deriva de $HOME en vez de hardcodear una ruta. Override: CONDA_SH.
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: no encuentro conda.sh en $CONDA_SH" >&2
    echo "       Corri 'conda info --base' en el login y exporta:" >&2
    echo "       export CONDA_SH=<base>/etc/profile.d/conda.sh" >&2
    exit 1
fi
source "$CONDA_SH"
# El env SI es compartido por todo el grupo (objetivo 3 del documento).
conda activate /data/contrib/pci_78/envs/nmr_xpu

# El wheel torch==2.13.0+xpu trae su propio runtime oneAPI: no hace falta
# `module load intel/2025.3.0` (verificado en cn073, Fase 0).

# --- Rutas y dispositivo --------------------------------------------------
export NMR_DATA_DIR="${NMR_DATA_DIR:-/data/contrib/pci_78/Lucas/DB_202K}"
export NMR_DEVICE=xpu     # exige XPU: si no la ve, aborta en vez de usar CPU

# Repo en el espacio de trabajo de cada usuario (no compartido).
NMR_REPO="${NMR_REPO:-$HOME/nmr-hsqc-to-vector-}"
cd "$NMR_REPO/experiments/E3_dos_conjuntos"

# --- Chequeo previo (regla 5: nunca gastar cola de GPU a ciegas) -----------
python -u tests/test_device_utils.py
python -u tests/test_config_utils.py
python -u tests/test_forward_settransformer.py
python -u tests/test_paridad_cpu_xpu.py    # paridad CPU<->XPU (Fase 2)

echo "--- Dispositivo visible para PyTorch ---"
python -u -c "import torch; print('torch', torch.__version__); print('xpu?', torch.xpu.is_available()); print('n_dev', torch.xpu.device_count())"

# --- Entrenamiento --------------------------------------------------------
python -u train.py --config config_settransformer.yaml
