# Tutorial: correr en Intel XPU (Clementina XXI)

> Guía operativa para cualquiera del grupo que quiera correr experimentos en
> las GPUs Intel de Clementina XXI. Para el "por qué" y las decisiones de
> diseño, ver [`MIGRACION_XPU_Clementina_XXI.md`](MIGRACION_XPU_Clementina_XXI.md).
> Todos los comandos de acá fueron **corridos y verificados** en `cn073`
> (2026-07-23), no son teóricos.
>
> ⚠ **Estado:** por ahora vive en la rama `feature/intel-xpu-support`. Una vez
> mergeada a `main`, cambiar el `-b feature/intel-xpu-support` del paso 1 por
> `main` (o quitarlo directamente).

---

## 0. Antes de arrancar

- Necesitás cuenta en Clementina XXI y acceso al espacio compartido
  `/data/contrib/pci_78`.
- El env de PyTorch-XPU ya está armado y congelado para todo el grupo en
  `/data/contrib/pci_78/envs/nmr_xpu` — no hay que instalar nada.
- Tu `conda` personal vive en tu `$HOME` (`~/miniconda3`), es individual por
  usuario. Lo compartido es el **env**, no el conda que lo activa.

## 1. Clonar el repo

Los nodos de Clementina no resuelven DNS externo directo: hace falta el proxy
del clúster antes de cualquier `git`/`pip`.

```bash
export http_proxy=172.28.3.3:3128
export https_proxy=172.28.3.3:3128

git clone -b feature/intel-xpu-support \
  https://github.com/Elpasu/nmr-hsqc-to-vector-.git \
  ~/nmr-hsqc-to-vector-
```

## 2. Pedir un nodo GPU interactivo (para probar, sin gastar cola)

```bash
srun -p gpunode --gres=gpu:intel_xt1550:1 --pty bash
```

Vas a esperar en cola un rato (fair-share); una vez asignado, entrás directo
al nodo.

## 3. Activar el entorno

```bash
unset ZE_AFFINITY_MASK                     # ver gotcha #1 mas abajo
conda activate /data/contrib/pci_78/envs/nmr_xpu
cd ~/nmr-hsqc-to-vector-/experiments/E3_dos_conjuntos
```

Verificación rápida de que PyTorch ve la GPU:

```bash
python -c "import torch; print('xpu?', torch.xpu.is_available()); print('n_dev', torch.xpu.device_count())"
```

Esperado: `xpu? True`, `n_dev 2` (los 2 tiles de la GPU asignada).

## 4. Correr los tests antes de tocar nada (regla del proyecto: nunca a ciegas)

```bash
python tests/test_device_utils.py       # abstraccion de dispositivo
python tests/test_config_utils.py       # rutas por variable de entorno
python tests/test_forward_settransformer.py   # shapes del modelo
python tests/test_paridad_cpu_xpu.py    # paridad numerica CPU<->XPU (tarda unos segundos)
```

Si los 4 dan `OK`, el entorno está sano y podés lanzar en serio.

## 5. Lanzar un entrenamiento real

Salí de la sesión interactiva (`exit`) y mandá el job por cola, desde el
nodo de login:

```bash
cd ~/nmr-hsqc-to-vector-/experiments/E3_dos_conjuntos
sbatch run_train_settransformer_clementina.sh
```

El script mismo corre los 4 tests del paso 4 como chequeo previo — si algo
falla, aborta antes de gastar horas de GPU.

Seguir el progreso:

```bash
squeue --me
tail -f expE3_st_train_xpu_<jobid>.out
```

Cuando termine, evaluar:

```bash
sbatch run_eval_clementina.sh config_settransformer.yaml
```

## 6. Correr otro experimento / config propio

Los `.sh` de Clementina son específicos de E3-SetTransformer (por ahora, es
el único experimento migrado — ver alcance en el documento de migración,
§5). Para adaptar el patrón a otro script:

- Copiá `run_train_settransformer_clementina.sh` como base.
- Cambiá el `python -u train.py --config ...` de la última línea.
- Si tu config no usa `${NMR_DATA_DIR:-...}` / `${NMR_DEVICE:-...}` en
  `paths.base_dir` / `system.device`, agregalo (ver
  [`config_utils.py`](../experiments/E3_dos_conjuntos/config_utils.py)) para
  que el mismo YAML sirva en login-1 y en Clementina sin duplicarlo.

---

## Gotchas (todos reales, todos ya nos pasaron)

**1. `ZE_AFFINITY_MASK` mal seteado por SLURM.** Sin `unset ZE_AFFINITY_MASK`
al arrancar, PyTorch puede no ver la GPU aunque el job la tenga asignada.
Ponerlo siempre, primera línea después del `#SBATCH`.

**2. `git pull`/`clone` falla con el env activado.**

```
/usr/libexec/git-core/git-remote-https: symbol lookup error: /lib64/libldap.so.2:
undefined symbol: EVP_md2, version OPENSSL_3.0.0
```

El env prioriza su propio OpenSSL y choca con el `git` del sistema. Fix,
sin desactivar el env:

```bash
LD_LIBRARY_PATH= git pull
```

**3. `set -euo pipefail` mata el job al activar conda.** Los hooks internos
de oneAPI (`mpivars.deactivate.sh`) referencian variables sin default
(`SETVARS_CALL`); con `set -u` activo, el job aborta con "unbound variable"
antes de llegar a `train.py`. Los `.sh` de Clementina ya tienen el fix (bajan
la guarda solo alrededor de `conda activate`); si escribís un `.sh` nuevo,
copiá ese bloque.

**4. Sin proxy no hay red.** `export http_proxy=172.28.3.3:3128` /
`https_proxy` antes de cualquier `git`/`pip` en un nodo de Clementina.

**5. Pedir un backend que no está disponible aborta, no cae a CPU en
silencio.** Si el config dice `device: xpu` (vía `NMR_DEVICE=xpu`) y el job
no ve la GPU, `pick_device()` tira `RuntimeError` en vez de entrenar horas
en CPU sin que nadie se entere. Es a propósito.
