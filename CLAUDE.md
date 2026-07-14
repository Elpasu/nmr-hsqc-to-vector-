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
   pending eterno con "Nodes DOWN/DRAINED".
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

---

## Los dos formatos de HSQC (no intercambiables)

- **V1 (1 canal)** `nmr_dataset_{N}.h5` → `hsqc (N, 256, 256)`. Modelos V7/V9.
- **V3 (2 canales)** `nmr_dataset_v3_{N}.h5` → `hsqc (N, 2, 256, 256)`. Modelos V8/V10/V11.
  Canal 0 = DEPT escalado por N_H; Canal 1 = tipo CH (0.33/0.67/1.0).

Todos los experimentos de este repo usan **2 canales (V3)**.

---

## Infraestructura

- **Cluster de entrenamiento:** login-1, user `lpassaglia.iquir`, env conda `NMR_env`,
  partición `gpua10_hi` (GPUs A10, 23 GB).
- **Cluster de datos (DFT/pkl/mapas):** snmgt01. Los datos ya están procesados; el
  pipeline de generación no forma parte del trabajo diario de este repo.
- **Datasets finales:** en `/home/lpassaglia.iquir/DB_200k` (login-1).

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
