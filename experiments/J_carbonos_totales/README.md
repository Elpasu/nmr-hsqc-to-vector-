# Exp J — Vector de carbonos totales

**Qué es:** el vector de 19 clases pasa a contar **carbonos**, no señales. El benceno da
`=CH/Ar = 6` en vez de 1, y `sum(vector) == C` de la fórmula molecular. La degeneración de cada
señal —derivada de la integración de protones— entra como 5ª feature de cada crosspeak.

**Qué NO es:** un reemplazo del checkpoint congelado. Exp J entrena un modelo para un target
distinto; los dos coexisten y sus EMAs **no son comparables**.

Spec: `docs/superpowers/specs/2026-08-12-vector-carbonos-totales-design.md`.

## Por qué

El vector viejo choca con la FM: dice 1 donde la fórmula dice C6. Para el generador de estructuras
aguas abajo hace falta el conteo real de carbonos. Afecta al **62 %** del dataset, con 3 carbonos
escondidos en promedio.

La integración lo resuelve: en el tolueno se ven 3 señales aromáticas CH, pero integran 2H / 2H / 1H
— de ahí salen los 5 carbonos aromáticos protonados.

## Las dos corridas

| Corrida | Config | `peak_features` | Qué mide |
|---|---|---|---|
| **J-A** | `config_j_a.yaml` | 5 | La propuesta completa (con degeneración) |
| **J-0** | `config_j_0.yaml` | 4 | Control: cuánto se logra **sin** integración |

**La diferencia entre las dos es el resultado.** Los dos configs leen los **mismos** archivos de
datos; J-0 solo recorta la 5ª columna. Si J-A ≈ J-0, la restricción de suma de la FM ya alcanzaba y
la integración es redundante — resultado igual de válido, y te ahorra pedirla en el laboratorio.

## Techo del experimento

Medido sobre 20 000 moléculas: **97,7 %**. Se descompone en 90,3 % de moléculas sin cuaternarios
escondidos (la integración resuelve toda la simetría) + 7,4 % donde los hay pero caen en una sola
clase Cq y la suma de la FM los ubica igual. Queda un 2,3 % genuinamente ambiguo.

Los cuaternarios no tienen integración: en un ¹³C real las intensidades no son cuantitativas.

## Cómo se corre

**1. Generar los datos (local, en la PC de Lucas — no en el cluster):**

Requiere que `vectors_13c_19v_202465.npy` (el ground truth existente, usado por el gate de
verificación) ya esté presente en `202K_suma/`.

```bash
python prep/make_labels_totales.py --config prep/config_prep.yaml
```

```bash
python prep/make_peaks_degeneracion.py --config prep/config_prep.yaml
```

El primero corre un **gate**: si el clasificador no reproduce exactamente los labels viejos, aborta
sin escribir nada. Genera `vectors_19v_totales_202465.npy` y `peaks_pkl_deg_202465.npz` en
`E:/Proyectos/SciTrix/ScitrixDB/DB_nmr_to_vector/202K_suma/`.

**2. Subir los dos archivos al cluster:**

```bash
scp vectors_19v_totales_202465.npy peaks_pkl_deg_202465.npz lpassaglia.iquir@login-1:/home/lpassaglia.iquir/DB_200k/
```

**3. Smoke tests antes de cualquier `sbatch` (regla dura 5):**

```bash
python tests/test_forward_j.py
```

```bash
python tests/test_dataset_j.py
```

```bash
python tests/test_configs_j.py
```

**4. Lanzar las dos corridas (desde `experiments/J_carbonos_totales/` en login-1):**

```bash
for c in config_j_a.yaml config_j_0.yaml; do sbatch run_train_j.sh "$c"; done
```

Cada job entrena Y evalúa, y deja todo en un `expJ_<jobid>.out`.

## Cluster

**login-1 / A10 ("capitán")**, env `NMR_env`, partición `gpua10_hi`, `base_dir`
`/home/lpassaglia.iquir/DB_200k`. Las dos corridas van al **mismo** cluster (regla dura 8) — mezclar
hardware metería la varianza A10-vs-XPU (~0,8 pp) dentro de la comparación.

No hay script para Clementina: el cupo QOS del grupo está trabado ahí, y Exp I ya ocupa esa cola.

## Métrica

**Primaria: EMA asistida v2** sobre el vector de carbonos, con `evaluate.py --oraculo all`.

> **Estos EMA no son comparables con el 92,14 % del checkpoint congelado.** Es otro target, más
> difícil por construcción: la suma promedio pasa de 11,4 a 13,3 y hay que acertar la degeneración
> además de la clase. El punto de comparación es **J-0**, no el modelo viejo.

## Tests

| Archivo | Qué verifica | Requiere |
|---|---|---|
| `prep/tests/test_labels_totales.py` | Benceno=6, tolueno, suma == C | rdkit, numpy |
| `prep/tests/test_peaks_degeneracion.py` | La degeneración cuenta en vez de descartar | rdkit, numpy |
| `tests/test_forward_j.py` | Forward con 4 y con 5 features (regla dura 5) | torch |
| `tests/test_dataset_j.py` | Normalización, recorte y `cond` | torch, rdkit |
| `tests/test_configs_j.py` | J-A y J-0 difieren solo en `peak_features` | PyYAML |
