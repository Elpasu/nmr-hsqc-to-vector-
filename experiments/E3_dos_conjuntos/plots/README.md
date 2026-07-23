# Plots — Exp E Fase 3 + estudio de escalado

Figuras generadas por `make_plots.py` a partir de los `.out` de train/eval y del
parquet de predicciones. **No depende de torch** — solo parsea texto y el parquet
(numpy + matplotlib + pandas/pyarrow).

## Cómo regenerarlas

```bash
cd experiments/E3_dos_conjuntos/plots
python make_plots.py
```

Por default lee los `.out` de `docs/runs/VE3_deepset+settransformers_HSQC+C/` y el
parquet de `docs/Runs/E3_settransformer/`. Se puede cambiar con `--runs-dir` y
`--parquet`. El script **auto-detecta** cada `.out` por el `experiment_name` de su
header (no por el jobid de SLURM), así que sirve aunque cambien los números de job.

## Figuras

| Archivo | Qué muestra |
|---|---|
| `train_curves_fase3.png` | Val/train loss vs época, DeepSets vs Set Transformer (Fase 3). |
| `train_curves_scaling.png` | Val loss vs época, las 5 fracciones de train (10/25/50/75/100%). |
| `ema_fase3.png` | EMA asistida vs baselines históricos (V10/Exp C/E2) + EMA cruda vs asistida. |
| `ema_por_entorno.png` | EMA asistida por entorno químico (sp3 / heteroátomos / sp2 / X-múltiples). |
| `confusion_topk_fase3.png` | Confusiones cruzadas (top-3 por clase, del `.out`), DeepSets \| Set Transformer. |
| `confusion_full_settransformer.png` | Matriz de confusión cruzada **completa** (recalculada desde el parquet). |
| `scaling_curve_ema.png` | Curva de escalado: EMA asistida y val loss vs tamaño de train (X en log). |

## Lecturas principales

- **Escalado en meseta:** de 75% a 100% del train la EMA asistida no sube (91.55% →
  91.35%) y el val loss es idéntico (0.0097) — ampliar el dataset otra vez no rendiría.
- **Set Transformer > DeepSets:** val loss ~2x menor (0.0097 vs 0.0201), EMA asistida
  91.35% vs 82.96%.
- **Confusión dominante restante:** `CH2`→`CH2-N` (181 señales en la matriz completa),
  seguida de `Imina`→`=CH/Ar` (101) y `C-2X`→`=CH/Ar` (70) — candidatos del próximo paso.
