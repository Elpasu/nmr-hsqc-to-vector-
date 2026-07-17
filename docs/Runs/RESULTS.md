# Results Log — NMR HSQC→Vector

One entry per run. Raw logs live in `docs/runs/<name>_train.out`.
**Target metric is EMA (Exact Match Accuracy), not Val Loss.** EMA comes from
`evaluate_v10.py`; Val Loss (MSE) only tracks convergence.

| Model | Ch | Cls | Data | Reg. | Best Val Loss (ep) | EMA crude | EMA assist | Notes |
|-------|----|----|------|------|--------------------|-----------|------------|-------|
| V10 baseline | 2 | 19 | 202k | none | 0.0303 (76) | 0.61% | 74.92% | overfits from ~ep48; assisted EMA inflated by oracle (Exp A) |
| V10-on-frozen-val (Exp D) | 2 | 19 | 202k | none | — (no retrain) | 0.93% | 90.66% | same ckpt as V10; val is ~90% train-contaminated, NOT a clean number — see below |

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

<!-- Template for next entries — keep it short:

## <Exp> — <one-line description>
- **Date:** · **SLURM:** · **Config:** · **Data:**
- **Change vs baseline:** <what this experiment tests>
- **Best Val Loss:** X @ epY · **EMA crude / assisted:** X / Y
- **Takeaway:** <1-2 lines>
- Also add a row to the top table.
-->
