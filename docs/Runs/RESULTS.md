# Results Log — NMR HSQC→Vector

One entry per run. Raw logs live in `docs/runs/<name>_train.out`.
**Target metric is EMA (Exact Match Accuracy), not Val Loss.** EMA comes from
`evaluate_v10.py`; Val Loss (MSE) only tracks convergence.

| Model | Ch | Cls | Data | Reg. | Best Val Loss (ep) | EMA crude | EMA assist | Notes |
|-------|----|----|------|------|--------------------|-----------|------------|-------|
| V10 baseline | 2 | 19 | 202k | none | 0.0303 (76) | 0.61% | 74.92% | overfits from ~ep48; assisted EMA inflated by oracle (Exp A) |
| V10-on-frozen-val (Exp D) | 2 | 19 | 202k | none | — (no retrain) | 0.93% | 90.66% | same ckpt as V10; val is ~90% train-contaminated, NOT a clean number — see below |
| Exp B — regularizacion | 2 | 19 | 202k | drop=0.25, wd=1e-5 | 0.1764 (97) | 0.00% | 27.09% | **regression, not fix** — underfits, see below |
| Exp C — GAP (fusion) | 2 | 19 | 202k | none | 0.0370 (100) | 0.89% | 70.02% | crude EMA improved vs V10 true baseline; assisted below target, see below |

---

## V10 — baseline (2ch + FM + 19v + 202k)

- **Date:** 2026-07-15 · **SLURM:** 2374453 · A10
- **Config:** `configs/config_V10.yaml` · **Data:** `nmr_dataset_v3_202465_fast.h5`
- **Arch:** `model_v10.py` (Conv2d 2→16→32→64 + 1D 512→256→128 + cond 8) — **no dropout / no weight_decay**
- **Sched:** ReduceLROnPlateau patience=8 factor=0.7 · **Split:** random_split(seed=42), val 10%
- **Run:** 100 epochs · 620 min (~6.1 min/ep) · `.err` clean · ckpt `checkpoints_V10_202k/nmr_202k_v10_2ch_fm_19v_best.pth`
- **Best Val Loss:** 0.0303 @ ep76
- **EMA:** cruda 0.61% · asistida 74.92% (Exp A, oráculo de doble restricción). Δ = +74.3pp — la EMA asistida satura la métrica, no sirve para comparar versiones. Ver `docs/PROMPT_superpowers_mejoras.md`.

**Takeaways:**
- Scheduler behaved correctly (LR 0.001 held to ep62, then smooth decay). No premature LR collapse (unlike V9).
- **Clear overfitting:** Train 0.013 vs Val 0.031; Val plateaus at ~0.031 from ep~48 while Train keeps dropping → validates **Exp B** (dropout + weight_decay). Epochs ~50–100 added no generalization.

---

## Exp D — val set congelado (V10 checkpoint, no retrain)

- **Date:** 2026-07-17 · **SLURM:** 2375430 · **Config:** `experiments/D_val_congelado/config.yaml`
- **Change vs baseline:** same V10 checkpoint (`nmr_202k_v10_2ch_fm_19v_best.pth`), re-evaluated on a
  frozen val (14428 "original 144k" molecules, historical `random_split(seed=42)` reproduced over
  `[0,144280)`), instead of the random 10%-of-202k split V10 was actually trained/evaluated on.
  `split.py` dedup report: 2928 canonical-duplicate groups (3433 excess molecules), 723 train rows
  dropped for leak against val, verified leak=0 (train∩val canonical SMILES).
- **EMA crude / assisted:** 0.93% / 90.66% (Δ +89.7pp). By entorno (assisted): Alifáticos 96.5%,
  Heteroatómicos O/N 95.0%, sp2 94.8%, X-Multiples 98.5%.
- **⚠️ Not a clean number:** ~90% of this frozen val was part of V10's actual training set (V10
  trained on a random 10%-of-202k split, a different permutation than this historical 144k-based
  split). The jump vs the original 0.61%/74.92% (Exp A, V10's true held-out split) is expected
  contamination, not a real improvement — treat this row as a rough anchor only. Exp B and Exp C
  will exclude this frozen val from their own training, so **their numbers on this same val ARE
  clean** — compare B vs C directly; compare either vs this row only with the caveat above in mind.
  See `experiments/D_val_congelado/RATIONALE.md`.
- **Takeaway:** split/dedup machinery verified working end-to-end (val landed exactly on the
  expected 14428, leak=0 confirmed). `val_indices_frozen.npy` now fixed for all future experiments.

---

## Exp B — regularización (dropout + weight_decay)

- **Date:** 2026-07-17→18 · **SLURM train:** 2375431 (100 ep, ~10.2h) · **SLURM eval:** 2376413
- **Config:** `experiments/B_regularizacion/config.yaml` · **Change vs baseline:** `model_v11b.py`
  (dropout=0.25 after ReLU of fc_fusion1/fc_fusion2) + weight_decay=1e-5 (Adam). Trained from
  scratch (not a re-eval like Exp D) on the Exp D frozen split (train=187314, val=14428, leak
  removed=723 — split machinery reproduced Exp D's numbers exactly, ruling out a split bug).
- **Loss:** Train 0.3745 / Val 0.1764 (best, ep97) — **~6-12x higher than V10's 0.013/0.031**, not
  in the same regime at all. LR decayed correctly (0.001→0.000058, no premature collapse). This
  is genuine underfitting, not measurement noise: 100 full epochs, LR annealed 17x, and train loss
  never got remotely close to V10's.
- **EMA crude / assisted:** 0.00% / 27.09% — both **worse than V10's true baseline** (0.61%/74.92%),
  not just worse than the inflated Exp-D reference. Δ=+27.09pp (much smaller than V10's +74.3pp,
  consistent with a model that's further from correct overall, so the oracle has less to work with).
- **Diagnosis (not yet confirmed, next step's job):** `fc_fusion1 = Linear(65664, 128)` is already
  a severe bottleneck — the entire HSQC image gets compressed into 128 numbers before dropout even
  applies. Stacking 25% dropout on that already-thin, overloaded channel (dropout again on the
  64-unit layer after it) likely starves the network of the image signal it needs, compounding the
  "modality collapse" Exp C already targets rather than fixing the overfitting gap. weight_decay=1e-5
  alone is too mild to explain a 6-12x loss inflation; dropout placement/magnitude is the prime
  suspect.
- **Takeaway:** regularization as specified (0.25 / 1e-5) is **not a safe default for this
  architecture** — do not carry these values into Exp C without re-testing. This result is
  evidence FOR prioritizing Exp C (rebalance the fusion bottleneck) over a milder regularization
  retry, since the failure mode implicates the same bottleneck Exp C targets.

---

## Exp C — GAP (rebalanceo de fusión) — resultado final

- **Date:** 2026-07-20→21 · **SLURM train:** 2376427 (100 ep, 603.7 min ≈ 10.1h) ·
  **SLURM eval:** 2376888 · **Params:** 222,883 (V10: 8,603,299, ~38.6x menos).
- **Loss:** Train 0.0203 / Val 0.0370 (mejor, ep100, todavía descendiendo — el scheduler
  nunca terminó de estabilizar). Mismo orden de magnitud que V10 (0.031); nada que ver con
  el 0.176 de Exp B. Ningún `[WARN]` de underfitting se disparó en las 100 épocas.
- **EMA crude / assisted:** 0.89% / 70.02% (Δ +69.1pp). Por entorno (asistida): Alifáticos
  82.71%, Heteroatómicos O/N 80.03%, sp2 83.54%, X-Multiples 96.32%.
- **Comparación honesta:** EMA cruda 0.89% > 0.61% de V10 (baseline real, split limpio) —
  mejora real, aunque chica en términos absolutos. EMA asistida 70.02% < 74.92% de V10 —
  por debajo, pero la asistida depende de dónde caen los errores del modelo (vía el
  oráculo), no es la métrica de comparación primaria del proyecto. Lejos del objetivo de
  ~90% asistida que se marcó el usuario — ningún experimento corrido hasta ahora se acercó
  a eso con una evaluación limpia.
- **Confusiones que sobreviven al oráculo — MISMAS que en V10 y Exp B:** `Cqsp2`↔`=CH/Ar`
  (~40% de los errores cruzados de esas dos clases), `CH2`↔`CH2-N` (44-52%), `CH`↔`=CH/Ar`.
  Tres arquitecturas distintas (V10 sin cambios, Exp B con regularización, Exp C con GAP),
  mismo patrón de confusión — evidencia fuerte de que el problema **no es arquitectónico**,
  es de representación/información. Motiva pasar a Exp E (representación de picos).
- **Takeaway:** rebalancear la fusión ayudó (crude EMA sube, 38.6x menos parámetros, sin
  underfitting) pero no resuelve las confusiones estructurales. Siguiente paso: Exp E
  (conjunto de picos en vez de imagen), no más iteración sobre la arquitectura CNN.

---

## Auditoría de distribución de clases (dataset completo, 202465 moléculas)

- **Fecha:** 2026-07-20 · **Script:** `scripts/audit_class_frequency.py` (login node, sin GPU).
- **Hipótesis original (descartada por los datos):** que `Cqsp2` y `=CH/Ar` fallan tanto en el
  modelo por ser clases *raras* en el dataset. Los datos dicen lo contrario.
- **Hallazgo real:** `Cqsp2` (92.62% de las moléculas, promedio 3.378/molécula) y `=CH/Ar` (80.34%,
  promedio 3.271/molécula) son las **dos clases más comunes de las 19**, no las más raras — están
  al fondo de la tabla ordenada por rareza, no arriba. Entre las dos suman ~6.6 señales promedio
  por molécula, una porción enorme del total de carbonos de una molécula típica.
- **Por qué importa:** como la EMA exige acertar las 19 clases simultáneamente, un error en la
  clase más frecuente y de mayor conteo arruina muchísimas más moléculas que un error en una clase
  rara. `Cqsp2` además es literalmente invisible en HSQC (carbono cuaternario, sin H propio) — el
  modelo solo puede inferirlo indirectamente (probablemente vía la Fórmula Molecular). La
  combinación "clase dominante + invisible en la imagen" la vuelve la sospechosa número uno de la
  EMA cruda baja, más que un problema aislado entre 19.
- **Bonus — composición de las 58k moléculas nuevas vs las 144k originales:** el dataset ampliado
  no es "más de lo mismo". Deltas de presencia más grandes: `CH2-N` 18.71%→57.34% (+38.6pp),
  `CH-N` 7.70%→33.78% (+26.1pp), `=CH/Ar` 76.57%→89.70% (+13.1pp), `Cqsp2` 90.79%→97.17% (+6.4pp).
  Las moléculas nuevas tienen mucho más contenido nitrogenado y más carbonos sp2/cuaternarios.
- **Implicancia para la estrategia:** el cuello de botella puede no ser solo arquitectónico
  (Exp C) — el desbalance de clases (`Cqsp2`/`=CH/Ar` dominan el conteo) sugiere que una loss que
  pondere por clase, o una cabeza de conteo específica para las clases dominantes, podría mover
  más la aguja que solo rebalancear la fusión. Ver discusión de próximos pasos en curso.

---

<!-- Template for next entries — keep it short:

## <Exp> — <one-line description>
- **Date:** · **SLURM:** · **Config:** · **Data:**
- **Change vs baseline:** <what this experiment tests>
- **Best Val Loss:** X @ epY · **EMA crude / assisted:** X / Y
- **Takeaway:** <1-2 lines>
- Also add a row to the top table.
-->
