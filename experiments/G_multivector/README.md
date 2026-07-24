# Exp G — Multi-vector (cobertura@K)

Generador de candidatos post-hoc sobre el checkpoint Fase 3 Set Transformer
(no reentrena). Emite hasta K vectores FM-consistentes por molecula; metrica =
cobertura@K sobre el val congelado.

## Piezas

- `candidates.py` — `generate_candidates(raw, total, ch2, n, o, K, max_swaps=2)`.
  Movimientos unitarios intra-grupo-de-nH desde el oraculo v2; todos FM-consistentes.
- `coverage.py` — curva cobertura@K sobre un parquet con `y_pred_raw`.
- `oraculo.py` — copia de E3 (reglas del ancla v2).
- `tests/` — numpy puro, corren local sin torch/GPU.

## Como correrlo

1. Tests locales (sin GPU):
   ```bash
   cd experiments/G_multivector
   python tests/test_candidates.py
   python tests/test_coverage.py
   ```
2. Generar el parquet con `y_pred_raw` en el cluster (checkpoint Fase 3 Set
   Transformer; XPU/Clementina o A10):
   ```bash
   cd experiments/E3_dos_conjuntos
   python dump_predictions.py --config config_settransformer.yaml
   ```
3. Traer el parquet a la PC y correr la metrica (100% local):
   ```bash
   cd experiments/G_multivector
   python coverage.py --parquet /ruta/al/predictions_nmr_202k_e3_settransformer_2sets_19v.parquet
   ```
4. Elegir el K operativo (~2-3) que da cobertura ~=98% con K promedio chico.
   Agregar la curva a `docs/Runs/RESULTS.md`.

## Alcance

v1 = solo intra-nH (techo 98.7%). Fase 2: candidatos cross-nH (el 1.28% de
multiplicidad mal) y/o reentrenar con distribucion calibrada por grupo de nH.
