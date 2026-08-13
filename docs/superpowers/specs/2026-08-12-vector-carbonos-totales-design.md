# Exp J — Vector de carbonos totales (simetría resuelta por integración)

**Fecha:** 2026-08-12
**Estado:** diseño aprobado, pendiente de plan de implementación
**Modelo base:** `experiments/E3_dos_conjuntos/` (Set Transformer, checkpoint congelado EMA v2 92.14 %).
El checkpoint congelado **no se toca**: Exp J entrena un modelo nuevo, con un target nuevo.

---

## 1. Motivación

El vector de 19 clases cuenta hoy **señales**, no carbonos. `Gen_vector.py` (y su puerto fiel
`smiles_classifier.py:102-110`) colapsa los carbonos equivalentes por simetría con
`CanonicalRankAtoms(breakTies=False)`, así que el benceno da `=CH/Ar = 1` y no 6.

Eso choca de frente con la Fórmula Molecular: la FM dice C6 y el vector suma 1. Para el generador de
estructuras aguas abajo el vector útil es el de **carbonos totales** — para armar una molécula hacen
falta todos los átomos, no una lista de entornos distintos.

**Cuánto importa** (medido sobre 20 000 moléculas del dataset, muestra aleatoria seed 42):

| Medición | Valor |
|---|---|
| Moléculas con simetría (carbonos > señales) | **62,0 %** |
| Carbonos escondidos, promedio en las que tienen simetría | **3,0** |
| Carbonos escondidos, promedio global | 1,85 |
| Máximo de carbonos escondidos en una molécula | 26 |
| Señales promedio / carbonos promedio | 11,42 / 13,27 |

No es un caso de borde: casi dos tercios del dataset está afectado.

**La idea de Lucas:** la integración de protones revela la degeneración. En el tolueno se ven 3
señales aromáticas CH, pero sus integrales son 2H (orto), 2H (meta) y 1H (para) — de ahí salen los 5
carbonos aromáticos protonados. La integración es un dato experimental estándar que hoy el modelo no
usa.

---

## 2. Viabilidad — está medida, no supuesta

De los carbonos escondidos por simetría, ¿cuántos son alcanzables por integración?

| | |
|---|---|
| Escondidos en carbonos **protonados** (integración los resuelve) | 33 777 → **91,4 %** |
| Escondidos en **cuaternarios** (sin integración posible) | 3 159 → 8,6 % |

Y por molécula, cuánto queda determinado por integración + FM:

| Caso | Fracción | Por qué |
|---|---|---|
| A — sin cuaternarios escondidos | **90,3 %** | La integración resuelve toda la simetría |
| B — cuaternarios escondidos, todos en **una sola** clase Cq | **7,4 %** | La suma de la FM ubica el resto sin ambigüedad |
| C — cuaternarios escondidos repartidos en **varias** clases Cq | 2,3 % | Ambigüedad real |

**Techo teórico del vector de carbonos totales: 97,7 %** (A + B).

El 2,3 % restante es exactamente el tipo de caso que la generación multi-vector (Exp G, Fase 1b)
maneja emitiendo un par de candidatos — pero eso queda **fuera de alcance** acá (§8).

**Por qué los cuaternarios no se pueden resolver:** en un ¹³C real las intensidades no son
cuantitativas (NOE, tiempos de relajación distintos). No hay integración de la que sacar la
degeneración de un carbono sin H. Lo único que los acota es la suma total de la FM.

---

## 3. Supuestos verificados antes de escribir este diseño

Los tres se comprobaron ejecutando código contra los datos reales, no se asumieron:

1. **El clasificador portado reproduce los labels actuales.** `smiles_classifier.true_vector_from_smiles`
   vs `vectors_13c_19v_202465.npy`: **100,000 % de coincidencia exacta sobre 5 000 moléculas, 0
   discrepancias.** Sin esto, quitarle el colapso de simetría sería construir sobre arena (regla
   dura 7: un desalineamiento de labels entrena basura en silencio).
2. **El vector sin colapso suma exactamente C.** 100,000 % sobre 5 000 moléculas, 0 fallos.
3. **Casos conocidos dan lo esperado:**
   - benceno `c1ccccc1` (C6) → `=CH/Ar: 6`
   - tolueno `Cc1ccccc1` (C7) → `CH3: 1, =CH/Ar: 5, Cqsp2: 1`
   - isopropanol `CC(C)O` (C3) → `CH3: 2, CH-O: 1`

---

## 4. Ubicación en el repo

Carpeta nueva **`experiments/J_carbonos_totales/`**, autocontenida (copiar, no importar), siguiendo
la convención estándar del proyecto.

Acá **sí** corresponde copiar, al revés que en el Exp I: los labels, el dataset y la semántica del
condicionante divergen de verdad respecto del E3. No es una variante bit-a-bit del código de
referencia, es un experimento con otro target.

```
experiments/J_carbonos_totales/
  prep/
    make_labels_totales.py       # labels sin colapso de simetria
    make_peaks_degeneracion.py   # crosspeaks + 5a feature
    config_prep.yaml             # rutas locales (Windows), no del cluster
  dataset_j.py                   # copia de dataset_e3.py + 5a feature + cond nuevo
  model_j_settransformer.py      # copia de model_e3_settransformer.py, proj_ch 4->5
  train.py                       # copia de E3/train.py
  evaluate.py                    # copia de E3/evaluate.py
  oraculo.py                     # copia SIN cambios de E3/oraculo.py
  split_utils.py, config_utils.py, device_utils.py   # copias sin cambios
  config_j_a.yaml                # corrida J-A (con degeneracion)
  config_j_0.yaml                # corrida J-0 (control, sin degeneracion)
  run_train_j.sh                 # login-1 / A10 (train + eval en un job)
  tests/...
  README.md · RATIONALE.md
```

**Cluster objetivo: login-1 / A10 ("capitán")**, `lpassaglia.iquir`, env `NMR_env`, partición
`gpua10_hi`. Es el que el diseño ya recomendaba (§8.1) y donde vive toda la serie histórica. No se
genera script para Clementina XXI: el cupo QOS del grupo está trabado ahí y sumar un `.sh` que no se
va a usar es código muerto. Si en algún momento hace falta, se clona el patrón ya probado de
`run_train_settransformer_clementina.sh`.

---

## 5. Los tres cambios de datos

Los tres se generan **localmente** (los pkl, SMILES y labels están en
`E:/Proyectos/SciTrix/ScitrixDB/DB_nmr_to_vector/`). No hace falta el cluster para prepararlos.

### 5.1 Labels — sin colapso de simetría

Mismo `classify_carbon` (copia fiel, ya verificada §3.1), pero recorriendo **todos** los carbonos en
vez de un representante por clase de equivalencia. Es literalmente quitar `CanonicalRankAtoms` y el
`seen` de `smiles_classifier.py:102-110`.

Salida: `vectors_19v_totales_202465.npy` (nombre nuevo — **no** pisar el archivo original, que sigue
siendo el ground truth del checkpoint congelado).

Propiedad garantizada: `sum(vector) == C` de la FM, para las 202 465.

### 5.2 Crosspeaks — 5ª feature: degeneración

`extract_peaks_pkl.py:24` (`_dedupe_symmetric_peaks`) ya agrupa los picos por (δC, δH) redondeado a 6
decimales y **descarta** los duplicados. El tamaño del grupo **es** la degeneración: el dato ya estaba
en el pipeline y se tiraba.

Feature nueva por crosspeak: `degeneracion` = nº de carbonos que comparten esa señal.

Salida: `peaks_pkl_deg_202465.npz`, con `peaks` de shape `(N, 32, 5)`.

**Distribución medida** (77 395 señales protonadas sobre 10 000 moléculas):

| Degeneración | Fracción |
|---|---|
| 1 | 80,5 % |
| 2 | 18,1 % |
| 3 | 0,74 % |
| 4 | 0,58 % |
| ≥6 | 0,08 % |

Máximo observado: 12. Normalización: `degeneracion / degeneracion_scale`, con
`degeneracion_scale: 4.0` en la sección `normalization` del config (regla dura 3: nada hardcodeado).

**Nota de fidelidad experimental:** el agrupamiento es por coincidencia de shift, no por simetría de
RDKit. Dos carbonos químicamente distintos con shifts accidentalmente iguales cuentan como
degeneración 2. Eso es **correcto**: en un espectro real esa coincidencia es indistinguible de la
simetría, se ve una sola señal con el doble de integral. Es la misma colisión del 2,19 % ya
documentada en Fase 1b, y el modelo tiene que aprender a convivir con ella.

### 5.3 Picos ¹³C — sin cambios

`peaks_13c_202465.npz` se reusa tal cual. Los cuaternarios **no** llevan degeneración, por lo
explicado en §2: no es un dato que exista en un espectro real.

**Cambia una validación:** hasta ahora `extract_peaks_13c_pkl.py` verificaba
`total_label == n_picos_13C` (~100 %). Con el vector nuevo `total_label` pasa a ser C, que es **mayor
o igual** que el número de picos. La validación correcta pasa a ser `n_picos_13C <= sum(label)`, y la
diferencia es justamente el número de carbonos escondidos por simetría.

---

## 6. Cambios de modelo y condicionante

Delta mínimo sobre el E3 — un solo cambio de arquitectura:

| Componente | Cambio |
|---|---|
| `model_j_settransformer.py` | `self.proj_ch = nn.Linear(4, d_model)` → `nn.Linear(peak_features, d_model)`, con `peak_features` leído de `cfg['model']` (default 5). |
| `dataset_j.py` | Lee la 5ª feature y la normaliza con `degeneracion_scale`; recorta a `peak_features` columnas. |
| `cond` (8 valores) | **Misma forma, semántica nueva** en dos slots. |
| `oraculo.py` | **Sin cambios.** Recibe números distintos y funciona igual. |

`peak_features` configurable es lo que permite que **un solo archivo de modelo** sirva para las dos
corridas: J-A lo pone en 5 y J-0 en 4. Sin eso harían falta dos modelos casi idénticos, que es
exactamente el tipo de copia que se desincroniza sin avisar.

### 6.1 El condicionante

| Slot | Antes | Ahora |
|---|---|---|
| 0 | nº de señales ¹³C | **C de la FM** (= suma del vector) |
| 1 | nº de señales CH2 | **carbonos CH2 totales** |
| 2-7 | C, H, N, O, S, Hal | igual |

Como antes, ambos se derivan del label en `dataset_j.py::__getitem__` (`sum(target)` y las 4 clases
de `IDX_CH2`). El slot 0 queda numéricamente redundante con el slot 2 (C); se deja igual para
mantener la forma del tensor y el delta de código mínimo.

**Los dos siguen siendo experimentalmente observables**, que es la condición para que el oráculo sea
legítimo y no una fuga:
- `total` = C de la FM. Antes lo contabas del espectro y si dos señales se solapaban lo contabas mal;
  ahora sale de la FM, **exacto y sin error de lectura**. Es una mejora de robustez sobre el esquema
  actual, no solo un cambio de target.
- `ch2` = Σ(integrales de los crosspeaks de fase negativa) / 2, leído del HSQC editado.

### 6.2 Por qué degeneración y no la integral cruda

El químico carga **H** (2, 3, 6…), que es lo que lee del ¹H integrado y normalizado. El pipeline
divide por los H-por-carbono —que ya conoce de la multiplicidad— y le pasa al modelo el **número de
carbonos equivalentes**.

Hacer la división afuera es inyectar conocimiento químico explícito como condicionante, que es la
palanca que más rindió en toda la serie histórica (CH2, Fórmula Molecular). Una división no es una
operación natural para una red; no hay razón para hacérsela aprender.

En la preparación de datos la degeneración sale directo del tamaño del grupo (§5.2); la división
`H / mult` solo aparece del lado de la inferencia experimental, cuando llegue.

---

## 7. Las dos corridas

| Corrida | Config | `model.peak_features` | Labels | Features de crosspeak |
|---|---|---|---|---|
| **J-A** | `config_j_a.yaml` | `5` | carbonos totales | δC, δH, amp0, amp1, **degeneración** |
| **J-0** | `config_j_0.yaml` | `4` | carbonos totales | δC, δH, amp0, amp1 |

**La diferencia entre las dos es el resultado del experimento.** Los dos configs leen el **mismo**
`peaks_pkl_deg_202465.npz` y el **mismo** archivo de labels; J-0 simplemente recorta la 5ª columna vía
`peak_features: 4`. Así el único factor que cambia entre las corridas es la disponibilidad de la
integración — no hay dos archivos de datos que puedan desincronizarse.

Sin el control, si J-A funciona no se puede afirmar que la integración fue necesaria: podría ser que
la restricción de suma de la FM (que en el esquema nuevo es exacta) ya alcanzara sola.

### 7.1 Métrica y lectura del resultado

**Primaria:** EMA sobre el vector de carbonos, modo asistido v2. Se reportan los tres modos con
`evaluate.py --oraculo all`, sin tocar el script.

**Referencia física: techo 97,7 %** (§2).

> **Advertencia que va en `RESULTS.md` en grande:** estos EMA **no son comparables** con el 92,14 %
> del checkpoint congelado. Es otro target, más difícil por construcción — la suma promedio pasa de
> 11,4 a 13,3 y hay que acertar la degeneración además de la clase. El punto de comparación es J-0,
> no el modelo viejo.

Las tres lecturas posibles, todas válidas:

| Resultado | Conclusión |
|---|---|
| J-A ≫ J-0 | La integración es la que hace el trabajo. Diseño validado. |
| J-A ≈ J-0 | La suma exacta de la FM ya alcanzaba; la integración es redundante. Resultado útil: ahorra pedirla en el laboratorio. |
| Ambas bajas | El vector de carbonos es genuinamente más difícil. Se documenta y se para. |

---

## 8. Invariantes y alcance

### 8.1 Invariantes (regla dura 8) — para que J-A y J-0 sean comparables entre sí

- Val congelado `val_indices_frozen.npy` (14 428 moléculas) en las dos.
- 100 épocas, `ConstrainedMSELoss(lambda_sum=0.5)`, scheduler `patience=8/factor=0.7` (regla dura 6).
- `num_workers: 0` (regla dura 1), `seed = 42`.
- 19 clases en el orden de `config/db.yaml` (regla dura 7).
- Dataset completo de 202 465, `train_fraction = 1.0`.
- **El mismo cluster para las dos: login-1 / A10.**

### 8.2 Dentro de alcance

Regeneración de labels y crosspeaks, la carpeta `J_carbonos_totales/` con sus copias y deltas, los
dos configs, los `.sh` para los dos clusters, los tests, y el README/RATIONALE. Todo listo para
`sbatch`.

### 8.3 Fuera de alcance

- **Lanzar los jobs.** Claude Code no puede hacer `sbatch` ni leer logs del cluster.
- **La GUI del Exp H.** La columna de integración y la inferencia con el modelo nuevo se agregan
  después de saber si el modelo aprende.
- **Multi-vector / τ sobre el vector nuevo** (el 2,3 % ambiguo del §2). Es la continuación natural,
  pero es otro experimento.
- **Reemplazar el checkpoint congelado.** Exp J produce un modelo para un target distinto; los dos
  coexisten.
- **El Exp I** (sweep de hiperparámetros) sigue su curso sin tocarlo.

---

## 9. Testing

Todo local, en CPU, **antes** de gastar cola de GPU (regla dura 5).

### 9.1 Validaciones de datos (sobre las 202 465 completas)

1. El clasificador con colapso reproduce `vectors_13c_19v_202465.npy` **exactamente**. Ya verificado
   al 100 % sobre 5 000 (§3.1); se corre sobre el total. **Si falla, se aborta el experimento** — el
   ground truth nuevo no sería confiable.
2. `sum(vector nuevo) == C` de la FM, para las 202 465.
3. `sum(vector nuevo) >= sum(vector viejo)`, y la diferencia == carbonos escondidos por simetría.
4. Consistencia de la degeneración: `Σ(degeneracion × mult)` sobre los crosspeaks **≤** nº de H sobre
   carbono de la molécula, con igualdad en las moléculas que tienen shift DFT para todos sus H.
   No es igualdad universal a propósito: `extract_peaks_from_pkl_molecule` descarta los carbonos sin
   shift en el pkl, así que un pkl incompleto da estrictamente menos. El test reporta **qué fracción
   cumple la igualdad** — si esa fracción es baja, hay un problema de datos que hay que mirar antes
   de entrenar, no un test que haya que relajar.
5. Alineación fila-por-fila de picos / labels / SMILES, reusando `verify_smiles_alignment` de
   `extract_peaks_pkl.py` (ya probada en Fase 1b).
6. `n_picos_13C <= sum(label)` en las 202 465 (§5.3).

### 9.2 Smoke tests de código

7. Forward del modelo con entrada de **5** features: shape `(B, 19)`, sin NaN, con máscaras mixtas
   incluida una molécula totalmente enmascarada.
8. Forward de la variante J-0 con **4** features (misma clase, `proj_ch` de 4).
9. `dataset_j.py` normaliza la 5ª columna con `degeneracion_scale` del config, y `cond[0] == C`,
   `cond[1] == carbonos CH2`.
10. Los dos configs difieren **solo** en `model.peak_features` (5 vs 4), `experiment_name` y
    `paths.checkpoint_dir`, y coinciden en todos los invariantes de §8.1 — incluidos los nombres de
    archivo de datos, que tienen que ser los mismos.

---

## 10. Presupuesto y logística

| Etapa | Costo |
|---|---|
| Regeneración local de labels + crosspeaks | ~5 min de CPU |
| J-A + J-0 | 2 × ~40 min ≈ **80 min de GPU** |

**Un paso manual de Lucas:** subir los datos nuevos al cluster.
`vectors_19v_totales_202465.npy` y `peaks_pkl_deg_202465.npz` se generan en la PC local y hay que
copiarlos por `scp` a `/home/lpassaglia.iquir/DB_200k/` en login-1 (el `base_dir` del config).

Exp J corre en **login-1 / A10**, así que no compite por la cola con los 23 jobs del Exp I, que están
encolados en Clementina esperando cupo QOS. Los dos experimentos avanzan en paralelo sin
interferirse.

---

## 11. Criterios de éxito

El experimento es exitoso si, al terminar, se puede afirmar con evidencia **cuál de las tres lecturas
de §7.1 es la correcta** — no si el número es alto.

Concretamente:

1. Los labels nuevos están verificados contra el ground truth existente (§9.1.1) y contra la FM
   (§9.1.2), así que el target es confiable.
2. J-A y J-0 corrieron con todos los invariantes de §8.1 idénticos, así que su diferencia es
   atribuible a la integración y nada más.
3. El resultado se lee contra el techo medido de 97,7 %, no contra un objetivo inventado ni contra el
   92,14 % (que mide otra cosa).
