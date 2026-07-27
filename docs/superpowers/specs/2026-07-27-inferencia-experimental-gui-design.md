# Diseño — Inferencia experimental con GUI (Exp H)

**Fecha:** 2026-07-27
**Autor:** Lucas Passaglia (con Claude Code)
**Estado:** propuesto (pendiente de revisión del usuario)

---

## 1. Motivación y decisión de contexto

El modelo E3 Set Transformer (checkpoint Clementina XPU, EMA asistido v2 **92.14%**, y con
oráculo v2 + Fase 1b **~97% de cobertura@K** con K adaptativo) **no se va a reentrenar**. Se
decide congelarlo y pasar a dos cosas:

- **(a) [este spec]** Probar la inferencia del vector sobre **datos experimentales** de moléculas
  conocidas, con una **interfaz** para cargar los picos a mano.
- **(b) [tarea siguiente, fuera de este spec]** Dejar la predicción lista para **acoplarla al
  generador de estructuras** (un `predict.py` fino que envuelve este mismo pipeline).

**Hallazgo que habilita todo:** el E3 **no consume la imagen HSQC**, consume **picos**:
- `peaks_ch` — crosspeaks C–H `(δC, δH, amp_ch0, amp_ch1)`, hasta 32.
- `peaks_13c` — picos ¹³C `(δC,)`, incluye cuaternarios.
- `cond` — `[total_señales, total_CH2, C, H, N, O, S, Hal]` (8 valores).

Un HSQC experimental da justamente crosspeaks y un ¹³C 1D da los carbonos. No hay que
reconstruir ninguna imagen: la inferencia experimental es un **adaptador de datos** (picos leídos
→ tensores en el formato exacto del modelo) + el forward + el post-proceso ya existente (oráculo
v2 + Fase 1b).

## 2. Qué es entrada real y qué es solo evaluación (distinción central)

En **uso real** (aguas abajo, alimentando el generador) NO se conoce la estructura. Lo que se
conoce es la **fórmula molecular** (de un HRMS) y el **espectro**. Por eso:

| dato | qué es | rol |
|---|---|---|
| **FM** (`C10H12N2O`) | fórmula molecular, del HRMS | **ENTRADA** (condicional) — reemplaza al SMILES, que es *la respuesta buscada* |
| **δC** | desplazamiento ¹³C del carbono | **ENTRADA** |
| **δH** | desplazamiento ¹H (vacío si carbono sin H) | **ENTRADA** |
| **`mult`** (CH/CH2/CH3/Cq) | multiplicidad = nº de H del carbono; se **lee del espectro** (HSQC editado da el signo CH2 vs CH/CH3; DEPT distingue CH de CH3) | **ENTRADA** — arma `amp_ch0/amp_ch1` |
| **`clase`** (1 de 19) | la clase con heteroátomo (CH2 vs CH2-N, etc.) | **SOLO EVALUACIÓN** (`y_true`); es *lo que la red predice*, NO una entrada |

La columna `clase` es **opcional**:
- **Con `clase`** → modo evaluación (moléculas conocidas): computa `y_true`, cobertura@K y qué
  pico puntual se confunde.
- **Sin `clase`** → modo predicción real: solo emite los K vectores candidatos (germen de (b)).

El `total_señales` y `total_CH2` del condicional se derivan del **propio espectro**, no del vector
verdadero: `total_señales` = nº de picos ¹³C (nº de filas) y `total_CH2` = nº de filas con
`mult == CH2`. La FM (`C,H,N,O,S,Hal`) se parsea del string. Así el condicional se arma sin filtrar
la respuesta.

## 3. Mapeo de datos (derivado de `extract_peaks_pkl.py` y `config/db.yaml`)

**19 clases (orden fijo de `db.yaml`, no reordenar):**
`0 CH3, 1 CH2, 2 CH, 3 Cq, 4 CH3-O, 5 CH2-O, 6 CH-O, 7 Cq-O, 8 CH3-N, 9 CH2-N, 10 CH-N,
11 Cq-N, 12 =CH2, 13 =CH/Ar, 14 Cqsp2, 15 Aldeh, 16 Imina, 17 C-2X, 18 C-3X`.

**Multiplicidad por nº de H (grupos de nH):**
- `CH3` (3H) → clases {0,4,8}
- `CH2` (2H) → clases {1,5,9,12}  (`=CH2` cuenta como CH2)  ← **índice CH2 del oráculo**
- `CH`  (1H) → clases {2,6,10,13,15,16}  (`=CH/Ar`, `Aldeh`, `Imina` cuentan como CH)
- `Cq`  (0H) → clases {3,7,11,14,17,18}

**Amplitudes del crosspeak (idéntico a `extract_peaks_pkl.extract_peaks_from_pkl_molecule`):**
sea `mult` ∈ {1,2,3} el nº de H (Cq→sin crosspeak):
- `phase = -1.0 if mult == 2 else 1.0`
- `amp_ch0 = phase * mult`   (CH3=+3, CH=+1, **CH2=−2**)
- `amp_ch1 = mult / 3.0`     (0.33 / 0.67 / 1.0)

**Construcción de los dos conjuntos por molécula:**
- `peaks_13c` ← **todas** las filas (protonadas y Cq): `(δC,)`.
- `peaks_ch` ← filas con `mult ∈ {CH,CH2,CH3}`: `(δC, δH, amp_ch0, amp_ch1)`.

**Normalización (de `config_settransformer.yaml`, no hardcodear):**
`δC → δC/220`, `δH → (δH − (−1))/(15 − (−1)) = (δH+1)/16`, `amp_ch0 → amp_ch0/3.0`,
`amp_ch1` tal cual, `peaks_13c: δC → δC/220`. Padding a `(32, 4)` con máscara para `peaks_ch`;
`(M,)` con máscara para `peaks_13c` (M = máx de picos ¹³C).

**Condicional (orden exacto):** `[total_señales, total_CH2, C, H, N, O, S, Hal]`.
`total_señales = nº filas`; `total_CH2 = nº filas con mult==CH2`; `C,H,N,O,S,Hal` del parseo de la
FM (Hal = F+Cl+Br+I; el espacio químico actual es CHON, así que S y Hal serán 0, pero se parsean
igual por generalidad).

**Vector verdadero (solo si hay `clase`):** histograma de las clases de las filas → `(19,)`.

## 4. Arquitectura (lógica separada de la GUI)

Carpeta autocontenida `experiments/H_inferencia_experimental/`. Corre **local** (como
`gui_inspector.py`), no en el cluster.

```
FM (string) + tabla de picos (GUI / data_editor)
   → adapter.py     (numpy puro): parse_formula, build_inputs, true_vector
   → predict_core.py (torch CPU): load_model+checkpoint (cache), predict_raw → raw(19)
                                   candidatos = oraculo v2 + generate_candidates_uncertainty (Fase 1b)
   → app_inferencia.py (Streamlit): render de K vectores + crudo + ancla + (si clase) cobertura/confusión
```

### 4.1 `adapter.py` — numpy puro, testeable local (sin torch, sin GUI)
- `parse_formula(formula: str) -> dict{C,H,N,O,S,Hal}` — regex sobre el string; elementos ausentes = 0.
- `MULT_H = {"CH":1, "CH2":2, "CH3":3, "Cq":0}`; `CH2_CLASS_IDX = [1,5,9,12]`.
- `build_inputs(peaks: list[dict], formula: dict, norm_cfg) -> (peaks_ch, mask_ch, peaks_13c, mask_13c, cond)`
  donde cada `peak` = `{"delta_c", "delta_h"|None, "mult"}`. Arma amps, normaliza, paddea.
- `true_vector(peaks_con_clase: list[dict]) -> np.ndarray(19,)` — histograma; error claro si una
  fila no trae `clase` en modo evaluación.
- Validaciones: `mult` válido; `δH` presente sii `mult != Cq`; FM parseable; ≥1 fila.

### 4.2 `predict_core.py` — torch (CPU)
- `load_model(checkpoint_path, cfg) -> NMR_SetTransformer` (importado de
  `E3_dos_conjuntos/model_e3_settransformer.py`); `load_state_dict(torch.load(path,
  map_location="cpu"))`; `model.eval()`. Cacheable por la GUI (`st.cache_resource`).
- `predict_raw(model, inputs) -> np.ndarray(19,)` — forward con batch=1, `torch.no_grad()`,
  devuelve el crudo (pre-redondeo), como la columna `y_pred_raw` de la Fase 1.
- `candidatos(raw, formula, total, ch2, tau, K_max) -> list[np.ndarray]` — usa `oraculo.py`
  (ancla v2, `ajustar_conteo_hetero`) y `generate_candidates_uncertainty` de
  `G_multivector/candidates.py`. `n_atoms=N`, `o_atoms=O` del parseo de FM.

### 4.3 `app_inferencia.py` — Streamlit (I/O fino, sin lógica de negocio)
- `st.text_input("Fórmula molecular")`.
- `st.data_editor` — tabla editable con columnas `δC`, `δH`, `mult` (selectbox
  CH/CH2/CH3/Cq), `clase` (selectbox de las 19 + vacío = opcional). Permite pegar desde Excel.
- `st.slider` para `τ` y `K_max` (defaults τ=1.5, K_max=6, punto de operación de Fase 1b).
- Botón **Predecir** → llama `adapter` + `predict_core` y renderiza:
  - la tabla de los **K vectores candidatos** (conteos por clase), el **crudo**, el **ancla** (v2).
  - si hay `clase`: `y_true`, si `y_true ∈ candidatos` (cubierto en K), y resaltado del/los pico(s)
    donde el ancla difiere del verdadero (qué confunde).

### 4.4 Reutilización vs. copia (desvío consciente de la convención)
La memoria del proyecto dice "carpetas de experimento autocontenidas: copiar, no importar". Acá se
hace un **desvío explícito y acotado**: se **importan** (vía `sys.path`) tres piezas ya estables y
testeadas — `model_e3_settransformer.py` (torch puro), `oraculo.py` (E3) y `candidates.py` (G) —
en vez de copiarlas. Motivo: duplicar el **oráculo** arriesga que las dos copias diverjan, lo que
por la **regla dura 7** corrompe labels en silencio. La única lógica *nueva* (adapter) sí vive
autocontenida y con sus tests.

## 5. Setup local (una vez, lo hace Lucas)
- `pip install torch --index-url https://download.pytorch.org/whl/cpu` (CPU; el resto —
  streamlit, numpy, rdkit, pandas — ya está para el inspector).
- `scp` del checkpoint desde Clementina:
  `/data/contrib/pci_78/Lucas/DB_202K/checkpoints_E3_settransformer/nmr_202k_e3_settransformer_2sets_19v_best.pth`
  → una ruta local. La app lee la ruta de una constante/campo configurable (portable, como el
  `PRED_FILE` del inspector: ruta hardcodeada si existe, si no relativa al repo).
- Lanzar: `streamlit run experiments/H_inferencia_experimental/app_inferencia.py`.

## 6. Testing / verificación
- **`adapter.py`** — tests numpy puros, corren local sin torch:
  - un `CH2` produce `amp_ch0 = −2`, `amp_ch1 = 0.67`; un `CH3` → `+3, 1.0`; un `Cq` no genera
    crosspeak pero sí pico ¹³C.
  - normalización coincide con el config (δC/220, δH=(δH+1)/16, amp/3).
  - `cond` correcto: `total_señales` = nº filas, `total_CH2` = nº `CH2`, FM parseada bien
    (incluye caso con N y O; caso sin heteroátomos).
  - `parse_formula` robusto: `C10H12N2O`, `CH4`, elementos ausentes = 0, dígito implícito 1.
  - `true_vector` = histograma; suma = nº filas.
- **`predict_core.py`** — smoke test cuando torch esté instalado: carga el checkpoint, forward de 1
  molécula de juguete, shape `(19,)`; `candidatos()` devuelve el ancla como primero y todos
  FM-consistentes.
- **Sanity de extremo a extremo:** cargar en la GUI los picos de **una molécula del val congelado**
  (tomados del parquet) y verificar que el vector predicho coincide con el `y_pred_assisted_v2` de
  esa fila del parquet — prueba que el adaptador reproduce el formato de entrenamiento.

## 7. Alcance

**Dentro:** `adapter.py`, `predict_core.py`, `app_inferencia.py`, tests de `adapter.py`,
`README.md`, y una molécula de ejemplo precargada.

**Fuera (tareas siguientes):**
- **(b)** `predict.py` — API fina `predict_vectors(picos, formula, tau, K_max) -> [vectores]` para
  el generador de estructuras (reusa `adapter` + `predict_core`, sin GUI).
- Peak-picking automático desde espectros crudos (Bruker/procesados). Hoy la entrada es la tabla
  leída a mano por la interfaz.
- Portar `Gen_vector.py` local para auto-computar `y_true` desde la estructura (hoy `y_true` sale
  de la columna `clase` anotada).

## 8. Criterios de éxito
- La GUI toma FM + tabla de picos de una molécula conocida y devuelve el/los vector(es) candidato(s).
- En modo evaluación, para una molécula del val congelado el vector predicho **reproduce** el
  `y_pred_assisted_v2` del parquet (el adaptador es fiel al formato de entrenamiento).
- `adapter.py` con tests locales en verde (numpy puro).
- Todo corre local (torch CPU), sin cluster.
