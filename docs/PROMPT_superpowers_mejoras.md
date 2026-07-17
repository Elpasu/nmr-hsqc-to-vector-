# Prompt para Claude Code + Superpowers — Proyecto de Mejoras del Modelo NMR

> Pegá esto en Claude Code (con Superpowers instalado) abierto en la raíz del repo
> `nmr-hsqc-to-vector`. Guardalo también como `docs/PROMPT_mejoras.md` para versionarlo.

---

## Contexto (leé esto primero, no generes nada todavía)

Antes de proponer o escribir cualquier cosa, leé estos archivos del repo y confirmá
que entendés el estado:
- `CLAUDE.md` (reglas duras, estado, infra)
- `config/db.yaml` (fuente de verdad: rutas, N=202465, 19 clases, hiperparámetros)
- `docs/WORKFLOW_V11_para_ClaudeCode.md` (los experimentos A→E ya especificados)
- `docs/runs/RESULTS.md` (resultados del V10 baseline y del Exp A)
- `docs/runs/V10/` (el dump de predicciones y el análisis de errores del V10)
- El código baseline: `src/models/model_v10.py`, `src/data/dataset_v10.py`,
  `src/train_v10.py`, `src/evaluate_v10.py`, `src/dump_predictions.py`

**Hallazgo central que motiva todo esto (del Exp A):**
- EMA **cruda** (modelo solo): **0.61%**
- EMA **asistida** (con oráculo de doble restricción): **74.92%**
- El oráculo inyecta dos observables reales del espectro (total de señales, total de
  CH2 en fase opuesta). Es un prior legítimo de un químico, PERO satura la métrica:
  aporta +74pp en todos los modelos por igual, así que la EMA asistida NO sirve para
  comparar versiones. **La métrica para comparar experimentos es la EMA cruda.**
- El V10 (202k, 2ch, FM, 19v) NO mejoró la EMA asistida frente a V7/V8/V9 pese a +40%
  de datos. Diagnóstico: (1) la métrica estaba saturada por el oráculo; (2) overfitting
  sin regularización (train 0.013 vs val 0.031, val plano desde ~ep48); (3) ramas de
  la red muy desbalanceadas (conv aporta 65536 features, 1D aporta 128, FM aporta 8 →
  la conv ahoga a las otras, "modality collapse"); (4) los 58k nuevos son scaffolds
  diversos (más difíciles), no "más de lo mismo".
- Confusiones que sobreviven al oráculo (del mapa de confusiones): CH2↔CH2-N,
  =CH/Ar↔Cqsp2, entornos O/N que se solapan. Ahí está el jugo real de mejora.

## Objetivo del proyecto de mejoras

Subir la **EMA cruda** (métrica primaria) y, secundariamente, la EMA asistida en las
confusiones que el oráculo no resuelve. Ejecutar los experimentos del workflow (A→E),
priorizando B (regularización) y C (balance de ramas), que atacan las causas 2 y 3.

## Qué quiero que produzcas

Un **proyecto de mejoras estructurado**, un experimento por vez, siguiendo el flujo de
Superpowers (brainstorm → plan → implementación test-first). Para CADA experimento, una
carpeta autocontenida bajo `experiments/<id>_<nombre>/` que contenga:

1. **`RATIONALE.md`** — fundamenta la prueba: qué hipótesis testea, qué causa del
   diagnóstico ataca, qué cambia exactamente respecto al baseline V10, qué métrica
   esperás mover y cuánto, y el criterio de éxito/fracaso. Corto y concreto (no más de
   1 página). Este archivo es obligatorio y va PRIMERO.
2. **Los scripts del modelo/dataset modificados** (solo lo que cambia respecto al V10;
   si algo no cambia, importarlo del baseline, no duplicarlo).
3. **`config.yaml`** del experimento (hereda de `config/db.yaml`, solo overrides).
4. **`train.py`** (o reuso del genérico parametrizado por config).
5. **`evaluate.py`** — evaluador con modo `both` (EMA cruda + asistida), idéntico en
   formato al `evaluate_v10.py` para que los resultados sean comparables.
6. **`dump_predictions.py`** — predictor que vuelca el parquet por molécula para la GUI
   (mismo formato que el del V10, para reusar `gui_inspector.py`).
7. **`run_train.sh`** y **`run_eval.sh`** — scripts SLURM listos para `sbatch`.
8. **`README.md`** — checklist de cómo correrlo en el cluster (orden de comandos).

## Reglas duras (de CLAUDE.md — innegociables, cada una es un bug ya cometido)

1. `num_workers: 0` en los DataLoader con h5py (o `worker_init_fn` fork-safe). Con >0
   deadlockea y la GPU queda al 0%.
2. SLURM: `#SBATCH --gres=gpu:1`, NUNCA `--gpus=1`.
3. Nada hardcodeado: rutas, nombres y constantes salen de `config/db.yaml`.
4. Los h5 deben tener `chunks=(1,2,256,256)` (un chunk por imagen), NUNCA `chunks=True`
   ni chunking automático (deja lecturas ~1300x más lentas sobre NFS, GPU al 0%). El
   dataset a usar es `nmr_dataset_v3_202465_fast.h5` (el rechunkeado), NO el original.
5. Smoke test (forward de 1 batch, sin checkpoint ni h5 real) obligatorio ANTES de
   proponer cualquier `sbatch`.
6. Scheduler `patience=8, factor=0.7`. No volver al agresivo.
7. `num_classes=19`, orden de clases fijo (el de `config/db.yaml`). Reordenar desalinea
   labels sin tirar error.
8. **Split idéntico entre experimentos** (mismo seed=42 y misma partición de val), o las
   EMAs no son comparables. Ver Exp D del workflow (val congelado) — implementarlo temprano.
9. Métrica primaria de comparación = **EMA cruda**. Reportar siempre las dos.

## Experimentos a implementar (orden y prioridad)

Del `docs/WORKFLOW_V11_para_ClaudeCode.md`. Resumen del qué y por qué:

- **Exp B — Regularización.** Reponer `dropout=0.25` (en fc_fusion1/fc_fusion2) y
  `weight_decay=1e-5` (en Adam). Ataca el overfitting medido. Cambio mínimo, alto valor.
  Criterio: baja el gap train/val; EMA cruda de val ≥ baseline.
- **Exp C — Balance de ramas (el de mayor impacto esperado).** Reemplazar el aplastado
  `view(-1, 65536) → Linear` por Global Average Pooling: `AdaptiveAvgPool2d(1)` → conv
  aporta 64 features en vez de 65536. Fusión balanceada: 64 + 128 + 8 = 200. Ataca el
  modality collapse. Criterio: nº de params baja ~10x; EMA cruda sube; MAE de Cqsp2 y
  =CH/Ar (que dependen de 1D/FM) mejora.
- **Exp D — Val set congelado.** Deduplicación interna por SMILES canónico (las 144k
  tenían ~3400 dups internos) + val fijo (las 14428 originales, Opción B del workflow)
  guardado en `val_indices_frozen.npy`. Todos los experimentos siguientes usan ese split.
  Ataca comparabilidad y fuga residual. Implementar temprano (antes o junto con B).
- **Exp E — Representación de picos como conjunto (exploratorio, futuro).** DeepSets o
  Set Transformer sobre tokens (δC, δH, mult) en vez de imagen. Requiere nueva
  representación de datos. Fase separada, después de B/C/D.

## Cómo quiero que trabajes

1. **Empezá con `/superpowers:brainstorm`** sobre el proyecto de mejoras completo, para
   alinear el enfoque conmigo antes de escribir nada.
2. Después, **un experimento por vez**, en este orden: **D primero** (para fijar el split
   comparable), luego **B**, luego **C**. E queda para una fase posterior.
3. Para cada uno: `/superpowers:write-plan` → me mostrás el plan y el `RATIONALE.md`
   ANTES de escribir código → apruebo → `/superpowers:execute-plan`.
4. No corras nada en el cluster (no tenés acceso). Dejá todo LISTO para que yo haga
   `git pull` en el cluster, corra el smoke test, y lance `sbatch`. El README de cada
   experimento tiene que decir exactamente qué comandos corro yo y en qué orden.
5. Cada experimento termina con su entrada lista para agregar a `docs/runs/RESULTS.md`
   (yo la completo con los números cuando corra en el cluster).

## Empezá ahora

Confirmá que leíste el contexto y arrancá con `/superpowers:brainstorm` del proyecto de
mejoras. No escribas archivos hasta que acordemos el enfoque y el orden.
