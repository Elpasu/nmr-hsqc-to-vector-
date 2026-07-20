# Results Log — NMR HSQC→Vector

One entry per run. Raw logs live in `docs/runs/<name>_train.out`.
**Target metric is EMA (Exact Match Accuracy), not Val Loss.** EMA comes from
`evaluate_v10.py`; Val Loss (MSE) only tracks convergence.

| Model | Ch | Cls | Data | Reg. | Best Val Loss (ep) | EMA crude | EMA assist | Notes |
|-------|----|----|------|------|--------------------|-----------|------------|-------|
| V10 baseline | 2 | 19 | 202k | none | 0.0303 (76) | 0.61% | 74.92% | overfits from ~ep48; assisted EMA inflated by oracle (Exp A) |
| V10-on-frozen-val (Exp D) | 2 | 19 | 202k | none | — (no retrain) | 0.93% | 90.66% | same ckpt as V10; val is ~90% train-contaminated, NOT a clean number — see below |
| Exp B — regularizacion | 2 | 19 | 202k | drop=0.25, wd=1e-5 | 0.1764 (97) | 0.00% | 27.09% | **regression, not fix** — underfits, see below |

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

<!-- Template for next entries — keep it short:

## <Exp> — <one-line description>
- **Date:** · **SLURM:** · **Config:** · **Data:**
- **Change vs baseline:** <what this experiment tests>
- **Best Val Loss:** X @ epY · **EMA crude / assisted:** X / Y
- **Takeaway:** <1-2 lines>
- Also add a row to the top table.
-->
