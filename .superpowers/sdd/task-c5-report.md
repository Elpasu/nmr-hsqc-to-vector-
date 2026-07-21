# Task 5 Report: `config.yaml`

**Date:** 2026-07-20
**Task:** Create `experiments/C_gap/config.yaml` per Exp C plan

## Status
✅ **DONE**

## What Was Done

1. Read task brief from `.superpowers/sdd/task-c5-brief.md`
2. Created `experiments/C_gap/config.yaml` with exact YAML content from brief
3. Committed with exact message: `"exp C: agrega config.yaml"`
4. Self-reviewed

## Files Changed

- **Created:** `experiments/C_gap/config.yaml` (33 lines)
- **Commit:** `20df92b exp C: agrega config.yaml`

## Self-Review Findings

### Content Verification
- YAML structure matches brief exactly:
  - `experiment_name`: `"nmr_202k_v11c_gap_2ch_fm_19v"`
  - All path keys present: `base_dir`, `h5_filename`, `labels_filename`, `smiles_filename`, `checkpoint_dir`, `val_indices_filename`
  - Hyperparameters: `batch_size: 64`, `lr: 0.001`, `epochs: 100`
  - Scheduler: `patience: 8, factor: 0.7` (project standard)
  - System: `device: cuda`, `num_workers: 0`, `pin_memory: true`

### Hard Rules Compliance
- ✓ `num_workers: 0` — respects h5py fork-safety rule (CLAUDE.md)
- ✓ Uses `val_indices_frozen.npy` — frozen split from Exp D (Global Constraints)
- ✓ NO `regularization:` section — deliberate for Exp C (Exp B's dropout/weight_decay caused underfitting)
- ✓ Uses `_fast` variant: `nmr_dataset_v3_202465_fast.h5` (correct vs. config_V10.yaml)

### Commit Message
- ✓ Exact match: `"exp C: agrega config.yaml"`

## No Issues or Concerns

File is ready for downstream tasks:
- Task 6: `train.py` will read this config
- Task 7: `evaluate.py` will read this config
- Task 8: `dump_predictions.py` will read this config
