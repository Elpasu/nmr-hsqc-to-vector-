# Exp I — estudio de hiperparámetros del Set Transformer

**Qué es:** 23 corridas controladas que varían los hiperparámetros de arquitectura y optimización
del Set Transformer del Exp E Fase 3, con el mismo val congelado y el mismo presupuesto de épocas,
para poder justificar la arquitectura elegida ante un revisor.

**Qué NO es:** una búsqueda para reemplazar el checkpoint congelado. Si aparece un ganador real
(mejora mayor a la banda de ruido medida), eso se documenta y se decide aparte.

Spec: `docs/superpowers/specs/2026-08-07-estudio-hiperparametros-e3-design.md`.

## Por qué vive acá y no en `experiments/I_.../`

El estudio solo vale si el código de entrenamiento y evaluación es **bit a bit el mismo** que produjo
el checkpoint congelado. Copiar `train.py` a otra carpeta abriría el riesgo de divergencia silenciosa
que invalidaría toda la comparación. Mismo criterio que el estudio de escalado de datos, que también
vivió en esta carpeta.

## El diseño

Baseline (config congelada): `d_model=64, n_heads=4, n_layers=2, n_seeds=1, lr=1e-3, bs=64,
fusion=128→64`.

| Bloque | Qué varía | Corridas |
|---|---|---|
| OFAT | `d_model` {32,128,256}, `n_layers` {1,3,4}, `n_heads` {2,8}, `n_seeds` {2,4}, `lr` {3e-4,3e-3}, `batch_size` {32,128}, fusión {64→32, 256→128} | 16 |
| Grid 2D | `d_model` × `n_layers`, las 4 celdas que el OFAT no cubre | 4 |
| Réplicas | mismo config, seeds 42/43/44 → mide el piso de ruido | 3 |

Todo lo demás es invariante en las 23 (regla dura 8): val congelado, 100 épocas,
`ConstrainedMSELoss`, scheduler `patience=8/factor=0.7`, `num_workers=0`, 19 clases, dataset completo.

**Correr las 23 en el mismo cluster.** Recomendado login-1/A10 (~39 min por corrida ⇒ ~15 h de GPU).

## Cómo se corre

**1. Antes de cualquier `sbatch` — smoke test local (regla dura 5):**

```bash
python hp_sweep/tests/test_make_configs.py
```

```bash
python hp_sweep/tests/test_all_archs_forward.py
```

`test_all_archs_forward.py` necesita `torch` instalado — correrlo en el env `NMR_env`
(en el login node alcanza: es CPU-only, no hace falta GPU). `test_make_configs.py` en
cambio solo necesita PyYAML y corre en cualquier lado, incluida una PC Windows sin torch.

**2. Lanzar las 23 (desde `experiments/E3_dos_conjuntos/` en el cluster):**

```bash
for cfg in hp_sweep/configs/*.yaml; do sbatch run_sweep.sh "$cfg"; done
```

Cada job entrena Y evalúa el mismo config, y deja todo en un `expE3_hp_<jobid>.out`.

**3. Bajar los `.out` a la PC local y recolectar:**

```bash
python hp_sweep/collect_results.py --out-dir ruta/a/los/out
```

Escribe `hp_sweep/results.csv` e imprime la tabla markdown lista para pegar en
`docs/Runs/RESULTS.md`. Notas:

- Si un job murió o fue matado (ej. TIME LIMIT), el traceback de Python o el mensaje
  de cancelación de SLURM va al `expE3_hp_<jobid>.err`, **no** al `.out` que lee
  `collect_results.py`. Una corrida que queda PENDIENTE indefinidamente se
  diagnostica revisando su `.err`.
- Si relanzás un job que falló, borrá su `.out` viejo antes de volver a recolectar —
  si no, `collect_results.py` tira un error (dos `.out` para la misma corrida).
- `collect_results.py` ahora completa PENDIENTE las corridas de las 23 que todavía no
  tienen `.out` (lee `hp_sweep/configs/` por default), así que la tabla siempre tiene
  exactamente 23 filas.

**4. Generar la figura:**

```bash
python hp_sweep/make_plot.py
```

Escribe `experiments/E3_dos_conjuntos/plots/hp_sweep_ofat.png`.

## Cómo se cambia el diseño

Editar `sweep_grid.yaml` (la fuente única) y regenerar:

```bash
python hp_sweep/make_configs.py
```

Los YAML de `configs/` son **generados**: editarlos a mano se pierde en la próxima regeneración.
Para verificar que están al día sin escribir nada:

```bash
python hp_sweep/make_configs.py --check
```

## Métrica de decisión

**Primaria: EMA asistida v2.** Desempate: best val loss. La EMA cruda se reporta pero **no decide**
— `RESULTS.md` ya documenta que es ruidosa (0.9–2.3 % a lo largo del scaling study, sin tendencia
monótona pese a que la asistida subía monótonamente).

**Regla fijada antes de ver los resultados:** una diferencia menor a la banda de ruido de las 3
réplicas se declara **no significativa**.

## Tests

| Archivo | Qué verifica | Requiere |
|---|---|---|
| `tests/test_make_configs.py` | Invariantes del diseño experimental (23 configs, nombres únicos, un solo eje por corrida OFAT, etc.) | PyYAML, corre en cualquier lado |
| `tests/test_all_archs_forward.py` | Regla dura 5: smoke test de forward pass de las 23 configs reales | `torch`, env `NMR_env` |
| `tests/test_collect_results.py` | Parsing de los `.out` de SLURM a filas de resultado | Python puro |
| `tests/test_make_plot.py` | Generación de la figura OFAT | `matplotlib` |
