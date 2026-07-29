# Exp H — Inferencia experimental sobre datos espectroscópicos locales

## Qué es

Interfaz **Streamlit local** para predecir el vector de conteos de 19 clases de entornos
de carbono (CH3, CH2, CH, Cq, ..., C-2X, C-3X) a partir de:

- **Datos experimentales:** tabla editadle in-situ con picos (δ¹³C, δ¹H, multiplicidad).
- **Fórmula molecular:** reemplaza al SMILES como entrada conocida.
- **Modelo congelado:** E3 SetTransformer (Exp E Fase 3, EMA v2 = 92.14%, cobertura Fase 1b ~97%).

**No necesita cluster.** Corre en tu PC (CPU o GPU local) con PyTorch.

---

## Setup (una sola vez)

### 1. Instalar PyTorch CPU

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

(Si tienes GPU NVIDIA disponible localmente, puedes omitir `--index-url` para usar `cuda`.)

### 2. Descargar el checkpoint

El modelo está en Clementina. Cópialo a una carpeta local:

```bash
scp lpassaglia.iquir@clementina.iquir.unlp.edu.ar:/data/contrib/pci_78/Lucas/DB_202K/checkpoints_E3_settransformer/nmr_202k_e3_settransformer_2sets_19v_best.pth \
  checkpoints_local/
```

Crea la carpeta si no existe:

```bash
mkdir -p checkpoints_local
```

**Nota:** El archivo es grande (~1.2 GB). Asegúrate de tener espacio en disco.

### 3. Confirmar la ruta del checkpoint

El app espera la ruta:
```
<repo>/checkpoints_local/nmr_202k_e3_settransformer_2sets_19v_best.pth
```

Si prefieres otra ubicación, edita `app_inferencia.py` línea 27:
```python
_HARDCODED = r"TU_RUTA_AQUI\nmr_202k_e3_settransformer_2sets_19v_best.pth"
```

---

## Uso

### Lanzar la interfaz

```bash
streamlit run experiments/H_inferencia_experimental/app_inferencia.py
```

Se abre automáticamente en `http://localhost:8501`.

### Formato de la tabla

Edita la tabla directamente en la GUI. Cada fila es un carbono.

| Columna | Tipo | Requerido | Descripción |
|---------|------|-----------|-------------|
| `delta_c` | float | Sí | Desplazamiento ¹³C en ppm (0–220) |
| `delta_h` | float | No* | Desplazamiento ¹H en ppm (−1–15) |
| `mult` | select | Sí | Multiplicidad vista en HSQC/DEPT: `CH3`, `CH2`, `CH`, `Cq` |
| `clase` | select | No** | Clase verdadera (solo para evaluar) |

*) **`delta_h` es obligatorio excepto para `Cq`** (carbonos sin protones):
  - CH3, CH2, CH → requieren δ¹H
  - Cq → δ¹H debe estar vacío

**) **`clase` es opcional.** La entrada real es `mult` (lo que ves en el espectro).

### Entrada vs. Evaluación

- **Entrada (siempre requerida):** `delta_c`, `delta_h`, `mult`
  - Datos crudos del espectro editado.
  - `mult` es inferido del HSQC/DEPT (Fase 1b).

- **Evaluación (opcional, dos formas):**
  - **SMILES (recomendado para inferencia controlada — molécula conocida):** un campo
    de texto arriba de la tabla. Si lo completás, el vector verdadero se **calcula
    automáticamente** desde la estructura (`smiles_classifier.py`, puerto fiel de
    `Gen_vector.py`, el mismo clasificador que generó las labels de entrenamiento) —
    no hace falta anotar `clase` fila por fila. Tiene prioridad sobre la columna `clase`.
  - **Columna `clase`** (respaldo si no tenés el SMILES a mano): anotás la clase
    verdadera de cada fila a mano. Solo se usa si el campo SMILES está vacío y
    **todas** las filas tienen `clase`.
  - Sin ninguna de las dos → modo predicción pura (no hay evaluación).

**Nota de fidelidad de `smiles_classifier.py`:** el clasificador original cuenta como
heteroátomo (para las clases C-2X/C-3X) *cualquier* vecino no-carbono, no solo N/O
(incluiría halógenos y S). El dataset de entrenamiento es CHON puro así que esa rama
nunca se ejerce en la práctica, pero si probás una molécula fuera de ese espacio químico
(con Cl, Br, S) el `y_true` calculado por SMILES puede no coincidir con lo que el
oráculo v2 asume (que solo mira N+O) — es una diferencia real entre "cómo se generaron
las labels" y "qué restricciones usa el oráculo para inferencia", no un bug de esta app.

### Parámetros de control

**τ (Fase 1b):** amplitud de la búsqueda de candidatos (default 1.5).
- Rango: 0.0–3.0
- Mayor τ → más candidatos (mayor cobertura, menos precisión).

**K_max:** límite de candidatos emitidos (default 6).
- Rango: 1–10
- Usado junto a τ para generar la lista final.

### Salida

- **Tabla de candidatos:** los K mejores vectores predichos, formateados por clase.
- **Crudo (redondeado):** el output bruto del modelo, antes de oráculo v2.
- **Evaluación (si hay `clase`):**
  - ✅ Verde: `y_true` está cubierto en los K candidatos.
  - ❌ Rojo: `y_true` NO está cubierto.
  - **Diferencia ancla v2 vs verdadero:** qué clases confunde el anclaje (candidato 0).

---

## Tests

### Ejecutar los tests

```bash
python experiments/H_inferencia_experimental/tests/test_adapter.py
python experiments/H_inferencia_experimental/tests/test_smiles_classifier.py
```

Salida esperada: `>>> 15 TESTS OK <<<` y `>>> 9 TESTS OK <<<` respectivamente.

`test_adapter.py` verifica:
- Parseo de fórmula molecular.
- Construcción de inputs (normalización, amplitudes, máscaras).
- Histogramas de clase (true_vector).
- Rechazo de entrada inválida.

`test_smiles_classifier.py` verifica el puerto de `Gen_vector.py` (clasificador de 19
clases desde SMILES, usado para el `y_true` automático): orden de clases alineado con
`classes_19v` de `db.yaml`, colapso por simetría (benceno → 1 solo entorno aromático),
las ramas Aldeh/Imina/C-2X/C-3X, y rechazo de SMILES inválido.

---

## Sanity check de extremo a extremo

### Vía test (automático)

El test `test_adapter.py` incluye un caso de verdad conocida: **etanol (C₂H₆O)**

```
CH₃ (δ¹³C 18.0, δ¹H 1.2, mult CH3) → clase CH3
CH₂-O (δ¹³C 58.0, δ¹H 3.7, mult CH2) → clase CH2-O
```

Vectorial esperado: `[1, 0, 0, 0, 0, 1, 0, 0, ...]` (1 CH3, 1 CH2-O).

El test verifica que `build_inputs()` + `true_vector()` reproducen estos valores exactamente.

**Ejecuta:**
```bash
python experiments/H_inferencia_experimental/tests/test_adapter.py
```

### Vía GUI (rápido, con el campo SMILES)

Confirmado en esta sesión con el checkpoint real: en la GUI, dejando `clase` vacío,
poné la fórmula `C2H6O`, cargá los dos picos del etanol (CH3 18.0/1.2, CH2 58.0/3.7) y
el SMILES `CCO` en el campo de arriba. El `y_true` se calcula solo (CH3 + CH2-O) y el
candidato 0 (ancla v2) lo reproduce exactamente — `y_true CUBIERTO en K ✅`.

### Vía GUI (con datos de validación)

Este chequeo confirma que el adaptador reproduce el formato de entrenamiento:
cargando los picos de una molécula del set de validación, el candidato 0 (ancla
v2) que emite la GUI debe **coincidir exactamente** con el `y_pred_assisted_v2`
de esa fila del parquet (ambos son el mismo oráculo v2 sobre la misma salida
cruda del modelo congelado).

**Ojo — la multiplicidad NO está en el parquet.** La columna `crosspeaks` guarda
solo `[δC, δH]`: las amplitudes se descartan al dumpear (ver `dump_predictions.py`,
`raw_peaks_ch[...][:, :2]`). La multiplicidad (`mult`) vive en las amplitudes del
archivo de picos de entrenamiento `peaks_pkl_202465.npz` (array `peaks`, columna 2
= `amp0`: CH3=+3, CH=+1, **CH2=−2**), que está en el cluster. Sin ese dato no se
puede reconstruir la entrada exacta desde el parquet solo. Dos formas de obtener
`mult` para el chequeo:

1. **Desde la estructura conocida** (la fila del parquet trae `smiles`): asignás a
   mano CH3/CH2/CH/Cq a cada carbono. Lo más simple para 1-2 moléculas.
2. **Desde el npz de entrenamiento** (cluster): `mult = |amp0|`, con signo negativo
   ⇒ CH2.

Pasos:

1. Elegí una fila del parquet
   `docs/Runs/E3_settransformer/predictions_nmr_202k_e3_settransformer_2sets_19v.parquet`
   (columnas `smiles`, `crosspeaks`, `c13_shifts`, `y_pred_assisted_v2`).
2. Fórmula molecular desde el SMILES (RDKit `CalcMolFormula`).
3. En la GUI, una fila por carbono: `delta_c` de `c13_shifts`; `delta_h` de la
   segunda coordenada de `crosspeaks[i]` (solo los protonados); `mult` según arriba;
   `clase` vacío.
4. Poné τ=0 y K_max=1: el candidato 0 debe reproducir `y_pred_assisted_v2` de esa
   fila. Si no coincide, el adaptador no es fiel al formato de entrenamiento —
   revisá normalización y amplitudes.

---

## Archivos principales

- `adapter.py` — Convierte picos + fórmula a tensores de entrada del E3.
- `predict_core.py` — Carga el checkpoint y ejecuta el forward (CPU).
- `app_inferencia.py` — Interfaz Streamlit.
- `tests/test_adapter.py` — Tests unitarios (15 casos).

---

## Requisitos

- Python 3.8+
- `torch` (CPU o GPU)
- `streamlit`
- `pandas`
- `numpy`
- `pyyaml`

Instala todo con:
```bash
pip install torch streamlit pandas pyyaml numpy --index-url https://download.pytorch.org/whl/cpu
```

---

## Troubleshooting

### "No encuentro el checkpoint"

Verifica que el archivo esté en:
```
<repo>/checkpoints_local/nmr_202k_e3_settransformer_2sets_19v_best.pth
```

O ajusta `_HARDCODED` en `app_inferencia.py`.

### "ValueError: mult invalido"

Usa solo: `CH3`, `CH2`, `CH`, `Cq`. Case-sensitive.

### "Entrada inválida"

- ¿Fórmula molecular válida? (ej. `C10H12N2O`, no `C10H12P`).
- ¿Todos los CH3/CH2/CH tienen `delta_h`?
- ¿Todos los Cq tienen `delta_h` vacío?

### Predicción diferente a la esperada

- Verifica τ y K_max (afectan los candidatos emitidos).
- Si ingresaste `clase`, asegúrate de que la clase verdadera sea exacta.
- Pequeñas diferencias (~1–2 átomos) son normales en moléculas complejas.

---

## Siguiente paso

**Exp H Parte (b):** `predict.py` — función `predict_vectors(picos, formula, tau, K_max)` para acople al generador de estructuras (sin GUI). Reutiliza `adapter` + `predict_core`.

---

**Autor:** Lucas Passaglia (UCA Team)  
**Fecha:** 2026-07-27  
**Modelo:** E3 SetTransformer (EMA v2 92.14%)
