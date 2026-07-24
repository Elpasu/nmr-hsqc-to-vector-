# CLAUDE.md — Contexto del proyecto NMR HSQC→Vector

> Este archivo lo lee Claude Code automáticamente al abrir el repo. Es el contexto
> permanente: qué es el proyecto, en qué estado está, y las reglas que NO se pueden
> romper. Para el detalle de los experimentos, ver `docs/WORKFLOW_V11_para_ClaudeCode.md`.
> Para rutas/constantes/nombres de archivo, ver `config/db.yaml` (fuente única de verdad).

---

## Qué es el proyecto

Modelo de deep learning que predice, a partir de un espectro **HSQC simulado** (imagen
2D) + proyecciones 1D (¹³C y ¹H) + un tensor condicional, un **vector de conteos enteros
de 19 clases** de entornos de carbono (CH3, CH2, CH, Cq, …, C-2X, C-3X).

- **Tarea:** predecir el vector *entero* correcto por molécula.
- **Métrica principal:** Exact Match Accuracy (EMA) — fracción de moléculas con el
  vector completo correcto (métrica exigente: un solo grupo mal ⇒ molécula cuenta como error).
- **Objetivo del proyecto:** EMA > 85% en validación hermética.
- **Autor:** Lucas Passaglia (UCA Team).

---

## Estado actual (2026-07-14)

- **V10 entrenando** en login-1 (2 canales + Fórmula Molecular + 19 clases + dataset 202k).
  Es el baseline actual y el candidato a superar el 85%.
- **Dataset ampliado a 202 465 moléculas** ya construido y validado
  (144 280 originales + 58 185 nuevas de scaffolds diversos). Anti-leak = 0 solapamientos.
  Vive en el cluster: `/home/lpassaglia.iquir/DB_200k`.
- **Próximo trabajo:** los experimentos A→E de `docs/WORKFLOW_V11_para_ClaudeCode.md`.
  Empezar por el A. Un experimento por vez.

**El código del V10 es la REFERENCIA — no se modifica.** Las variantes V11 se generan
por diferencia respecto a él.

> **Actualización 2026-07-24 — Exp E Fase 3 (Set Transformer) corre en dos clusters.**
> Además de login-1 (NVIDIA A10, CUDA), `experiments/E3_dos_conjuntos/` fue migrado y
> **validado** en Clementina XXI (Intel Data Center GPU Max 1550, backend `xpu`): mismo
> código, mismo config, resultado equivalente (de hecho mejor: EMA asistida 92.12% vs
> 91.35% baseline). Ver [Los dos clusters de entrenamiento](#los-dos-clusters-de-entrenamiento)
> más abajo y `docs/MIGRACION_XPU_Clementina_XXI.md`. **El resto de los experimentos
> (V10, B, C, E2, F, D) NO están migrados — siguen atados a CUDA/login-1**, es una
> decisión explícita de alcance, no un olvido.

---

## Serie histórica (para contexto, EMA en val hermético 144k)

V3 72.42 → V6-12clases 83.79 (máx histórico) → V7 83.36 → V8 (2ch sin FM) 78.51 →
V9 (FM+19v) 79.82 → **V10** (2ch+FM+19v+202k, en curso).

Lección de la serie: **inyectar conocimiento químico explícito como condicionante**
(CH2, Fórmula Molecular) fue la palanca más efectiva de mejora.

---

## Reglas duras (cada una viene de un bug real ya cometido — NO repetir)

1. **`num_workers: 0` con datasets h5py.** Con `num_workers > 0` el DataLoader
   deadlockea (h5py no es fork-safe): la GPU queda al 0% de util y el entrenamiento
   se cuelga sin tirar error. Ya pasó: 4h de GPU desperdiciadas. Alternativa avanzada:
   abrir el h5 por worker con `worker_init_fn` (fork-safe). Por defecto: 0 workers.
2. **SLURM usa `#SBATCH --gres=gpu:1`, NO `--gpus=1`.** Con `--gpus=1` el job queda
   pending eterno con "Nodes DOWN/DRAINED". (En Clementina/XPU es distinto:
   `--gres=gpu:intel_xt1550:1` — ver regla 9.)
3. **Nada hardcodeado.** Rutas, nombres de archivo y constantes salen SIEMPRE de
   `config/db.yaml`. (Bug real: `train_v9.py` tenía el `smiles_path` fijo, ignorando
   el config → riesgo de entrenar con la FM de las moléculas equivocadas.)
4. **Encoding UTF-8.** Al crear archivos en el cluster, heredoc con comillas
   (`cat > f.py << 'EOF'`). Si un `.py` tira `SyntaxError: invalid continuation byte`
   → encoding roto, arreglar con `iconv -f latin1 -t utf-8`.
5. **Smoke test obligatorio antes de cada `sbatch`.** Correr `tests/test_forward.py`
   (1 batch, en CPU/login node) y confirmar shapes. No gastar cola de GPU para
   descubrir un mismatch de dimensiones.
6. **Scheduler `patience=8, factor=0.7`.** Es el estándar del proyecto. NO volver al
   agresivo (`patience=3, factor=0.5`) que colapsó el LR prematuramente en V7/V9.
7. **`num_classes=19` y el orden de clases de `db.yaml` es fijo.** Reordenar las
   clases desalinea los labels con los espectros SIN tirar error → entrena basura en
   silencio.
8. **Comparabilidad:** val set y seed (42) idénticos entre experimentos, o las EMAs
   no son comparables. Ver Exp D (val set congelado).
9. **`set -u` en los `.sh` de Clementina/XPU rompe la activación de conda.** Los hooks
   internos de oneAPI (`mpivars.deactivate.sh`) referencian variables sin default
   (`SETVARS_CALL`); con `nounset` activo, el job aborta con "unbound variable" antes
   de llegar a `train.py`. Bajar la guarda (`set +u` / `set -u`) solo alrededor de
   `source conda.sh` + `conda activate`. Ver los `.sh` en `experiments/E3_dos_conjuntos/`.
10. **Windows no distingue mayúsculas/minúsculas en rutas; Linux (el cluster) sí.** Un
    `mkdir docs/runs/` descuidado en Windows, cuando el repo ya tiene `docs/Runs/`
    (mayúscula) trackeado, crea una carpeta indistinguible en Windows pero **otra
    carpeta completamente distinta** al clonar en el cluster. Antes de `git add` de un
    directorio nuevo, chequear `git ls-tree HEAD <ruta_padre>/` por si ya existe con
    otra capitalización.

---

## Los dos formatos de HSQC (no intercambiables)

- **V1 (1 canal)** `nmr_dataset_{N}.h5` → `hsqc (N, 256, 256)`. Modelos V7/V9.
- **V3 (2 canales)** `nmr_dataset_v3_{N}.h5` → `hsqc (N, 2, 256, 256)`. Modelos V8/V10/V11.
  Canal 0 = DEPT escalado por N_H; Canal 1 = tipo CH (0.33/0.67/1.0).

Todos los experimentos de este repo usan **2 canales (V3)**.

---

## Infraestructura

- **Cluster de entrenamiento (histórico, todos los experimentos):** login-1, user
  `lpassaglia.iquir`, env conda `NMR_env`, partición `gpua10_hi` (GPUs A10, 23 GB).
- **Cluster de datos (DFT/pkl/mapas):** snmgt01. Los datos ya están procesados; el
  pipeline de generación no forma parte del trabajo diario de este repo.
- **Datasets finales:** en `/home/lpassaglia.iquir/DB_200k` (login-1).

### Los dos clusters de entrenamiento (desde 2026-07-24, solo Exp E Fase 3)

| | login-1 (histórico, todos los exp.) | Clementina XXI (nuevo, **solo E3**) |
|---|---|---|
| GPU / backend | NVIDIA A10 / `cuda` | Intel Data Center GPU Max 1550 / `xpu` |
| Partición SLURM | `gpua10_hi`, `--gres=gpu:1` | `gpunode`, `--gres=gpu:intel_xt1550:1` |
| Datos | `/home/lpassaglia.iquir/DB_200k` | `/data/contrib/pci_78/Lucas/DB_202K` |
| Conda | `NMR_env` (prefijo del usuario) | `/data/contrib/pci_78/envs/nmr_xpu` (compartido) |
| `.sh` | `run_train_settransformer.sh`, `run_eval.sh` | `run_train_settransformer_clementina.sh`, `run_eval_clementina.sh` |

`experiments/E3_dos_conjuntos/` corre en ambos **sin tocar código**: `train.py`,
`evaluate.py` y `dump_predictions.py` usan `device_utils.pick_device()` (`cuda → xpu →
cpu`, config `system.device`) y `config_utils.load_config()` (expande `${VAR:-default}`
en `paths.base_dir` y `system.device`). Sin exportar nada, el comportamiento es el
histórico de login-1. Detalle completo: `docs/MIGRACION_XPU_Clementina_XXI.md` (por qué
y decisiones) y `docs/TUTORIAL_XPU_Clementina.md` (cómo usarlo, paso a paso).

**Para que un script nuevo de E3 sea igual de agnóstico:** importar `pick_device()` /
`wants_pin_memory()` / `synchronize()` de `device_utils.py` en vez de
`torch.device("cuda" if torch.cuda.is_available() else "cpu")`, y `load_config()` de
`config_utils.py` en vez de `yaml.safe_load()` directo. Si el script necesita una ruta
nueva del config, escribirla como `"${MI_VAR:-valor_por_defecto_de_login-1}"`.

**Los demás experimentos (V10, B, C, E2, F, D) NO están migrados** — su `device` y
`base_dir` siguen hardcodeados a CUDA/login-1. Es alcance explícito de la migración
(`docs/MIGRACION_XPU_Clementina_XXI.md` §5), no un olvido: no asumir que corren en
Clementina sin antes revisar si fueron migrados.

---

## Qué puede y qué no puede hacer Claude Code acá

- **Puede:** generar y refactorizar scripts (dataset, model, train, evaluate), armar
  configs, escribir los `.sh` de SLURM, crear tests.
- **No puede:** lanzar jobs de SLURM, ver GPUs, ni leer logs del cluster. Eso lo hace
  Lucas manualmente por SSH. Claude Code deja todo LISTO para `sbatch`.

---

## Piezas críticas que Lucas controla a mano (no delegar a ciegas)

- **`config/db.yaml`** — fuente de verdad; cualquier error acá se propaga a todo.
- **El split de datos (Exp D)** — un error de split no tira excepción, solo corrompe
  resultados en silencio (fuga de datos). Es el bug más caro del proyecto.
