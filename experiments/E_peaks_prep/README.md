# Exp E — Fase 1: Extracción y Validación de Picos

Checklist para correr esto en el cluster. Es solo procesamiento de datos —
no usa GPU, no hay `sbatch`, corre directo en el login node.

## Orden de comandos

1. `git pull` en tu clon del repo en el cluster.
2. `cd experiments/E_peaks_prep`
3. Confirmar que existen en `/home/lpassaglia.iquir/DB_200k/`:
   - `nmr_dataset_v3_202465_fast.h5`
   - `vectors_13c_19v_202465.npy`
4. Smoke test obligatorio (regla 5 de CLAUDE.md) — corre en segundos, no
   toca el h5 real:
   ```bash
   python tests/test_calibration.py
   python tests/test_blob_detect.py
   python tests/test_padding.py
   python tests/test_validation_report.py
   python tests/test_smoke.py
   ```
   Todos deben terminar en `>>> ... OK <<<`. Si `test_blob_detect.py` o
   `test_smoke.py` fallan por `ModuleNotFoundError: scipy`, instalar scipy
   en el env (`conda install scipy` o `pip install scipy` dentro de
   `NMR_env`) antes de seguir.
5. Extraer los picos del dataset completo (202465 moléculas, corre en el
   login node, esperá que imprima el progreso — no debería tardar más que
   varios minutos, es procesamiento de imágenes en CPU, no entrenamiento):
   ```bash
   python extract_peaks.py --config config.yaml
   ```
   Al final debería mostrar `max_peaks detectado` y las stats de picos por
   molécula, y guardar `peaks_202465.h5` en `DB_200k/`.
6. Correr la validación:
   ```bash
   python validate_peaks.py --config config.yaml
   ```
7. Copiar la salida completa (match exacto %, % con colisión, deficit
   promedio, ejemplos) a `docs/Runs/RESULTS.md`, sección nueva "Exp E Fase 1".
8. Avisá a Claude Code con los números — con eso decidimos si la extracción
   por blob-detection es lo suficientemente limpia para pasar a la Fase 2
   (armar y entrenar el modelo de conjuntos), o si hace falta el fallback de
   reprocesar desde el pkl original.

## Nota

Esta fase no entrena nada ni usa GPU — es la validación previa a comprometer
horas de entrenamiento en una representación que todavía no se probó.
