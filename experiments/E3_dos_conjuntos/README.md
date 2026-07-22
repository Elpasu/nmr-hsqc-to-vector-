# Exp E — Fase 3: dos conjuntos de picos (crosspeaks C-H + ¹³C)

Checklist para correr esto en el cluster. Se corren **dos arquitecturas**
(DeepSets y Set Transformer) sobre la misma pipeline de datos, para comparar.
Como en Exp E Fase 2, entrenan rápido (sin CNN) — probablemente cada una
termine las 100 épocas en bastante menos de una hora.

## Antes de empezar: generar y copiar el archivo de picos ¹³C

El conjunto de picos ¹³C (todos los carbonos, con cuaternarios) hay que
generarlo primero, en tu máquina local Windows (como Fase 1b):

```bash
cd experiments/E_peaks_prep
python extract_peaks_13c_pkl.py --config config_pkl.yaml
```

Esto crea `peaks_13c_202465.npz` en tu `202K_suma` local. **Revisá el
reporte de validación:** el "Match exacto (picos_13C == total_label)" debería
dar **~100%** (mucho mejor que el 97% de los crosspeaks, porque ahora incluye
los cuaternarios). Si da bajo (< 95%), el pkl no tiene shifts de algún tipo de
carbono — PARÁ y avisá antes de entrenar.

Después copiá `peaks_13c_202465.npz` a `/home/lpassaglia.iquir/DB_200k/` en el
cluster (vía `scp` o lo que uses).

## Orden de comandos

1. `git pull` en tu clon del repo en el cluster.
2. `cd experiments/E3_dos_conjuntos`
3. Confirmar que existen en `/home/lpassaglia.iquir/DB_200k/`:
   - `peaks_pkl_202465.npz` (crosspeaks, de Fase 1b)
   - `peaks_13c_202465.npz` (¹³C, recién copiado)
   - `vectors_13c_19v_202465.npy`, `smiles_202465.npy`
   - `val_indices_frozen.npy` (Exp D)
4. **Smoke tests obligatorios antes de cualquier `sbatch` (rule 5):**
   ```bash
   python tests/test_dataset_e3.py
   python tests/test_forward_deepsets.py
   python tests/test_forward_settransformer.py
   python tests/test_oraculo.py
   ```
   Los smoke de modelo imprimen el conteo de parámetros: DeepSets ~35k,
   Set Transformer ~70k (ambos chicos por diseño). Si alguno no es chico
   (< 200k), avisá antes de entrenar.
   > Nota: estos tests requieren torch y por eso se corren acá (login node),
   > no se pudieron correr en la máquina local de desarrollo (sin torch). El
   > único verificado localmente fue `experiments/E_peaks_prep/tests/test_extract_13c.py`.
5. Lanzar los dos entrenamientos:
   ```bash
   sbatch run_train_deepsets.sh
   sbatch run_train_settransformer.sh
   ```
6. **Revisá el log temprano** (como en Fase 2, entrena en minutos): mirá
   `expE3_ds_train_<jobid>.out` y `expE3_st_train_<jobid>.out`. Comparar el
   val loss final contra V10 (0.031), Exp C (0.037) y E2 (0.032).
7. Evaluar cada checkpoint sobre el val congelado:
   ```bash
   sbatch run_eval.sh config_deepsets.yaml
   sbatch run_eval.sh config_settransformer.yaml
   ```
8. (Opcional, para la GUI) Volcar predicciones:
   ```bash
   python dump_predictions.py --config config_deepsets.yaml
   python dump_predictions.py --config config_settransformer.yaml
   ```
9. Revisar los `expE3_eval_<jobid>.out`: copiar la tabla "EMA CRUDA vs
   ASISTIDA" y, sobre todo, mirar si las confusiones `Cqsp2`↔`=CH/Ar` y
   `CH2`↔`CH2-N` (idénticas en V10/B/C/E2) **bajaron** — es el indicador
   real de si agregar los cuaternarios resolvió el problema. Comparar contra
   Exp C (0.89% cruda) y E2 (0.74% cruda / 70.90% asistida).
10. Agregar dos filas a `docs/Runs/RESULTS.md` (una por arquitectura),
    fila "Exp E Fase 3".
11. Avisá a Claude Code con los números.

## Nota

Los dos modelos comparten dataset, loss, scheduler y split — solo cambia
`model.arch` en el config. La comparación DeepSets vs E2 aísla el efecto de
completar el input (agregar cuaternarios); Set Transformer vs DeepSets aísla
el de la capacidad relacional. Ver `RATIONALE.md`.
