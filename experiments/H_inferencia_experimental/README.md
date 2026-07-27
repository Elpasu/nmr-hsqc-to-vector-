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

- **Evaluación (opcional):** `clase`
  - Solo si conoces la clase verdadera de esa posición (moléculas de referencia).
  - Si todas las filas tienen `clase`, el app evalúa: ¿el vector predicho coincide con el verdadero?
  - Sin `clase` → modo predicción pura (no hay evaluación).

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

### Ejecutar los 15 tests del adaptador

```bash
python experiments/H_inferencia_experimental/tests/test_adapter.py
```

Salida esperada:
```
>>> 15 TESTS OK <<<
```

Los tests verifican:
- Parseo de fórmula molecular.
- Construcción de inputs (normalización, amplitudes, máscaras).
- Histogramas de clase (true_vector).
- Rechazo de entrada inválida.

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

### Vía GUI (con datos de validación)

1. **Toma una molécula del set de validación** del parquet:
   ```
   docs/Runs/E3_settransformer/predictions_nmr_202k_e3_settransformer_2sets_19v.parquet
   ```
   Columnas útiles: `smiles`, `crosspeaks`, `c13_shifts`, `y_pred_assisted_v2`.

2. **Extrae la fórmula molecular** del SMILES (ej. usando RDKit):
   ```python
   from rdkit import Chem
   mol = Chem.MolFromSmiles("CCCCCCCCC(CCCC)c1ccccc1")
   formula = Chem.rdMolDescriptors.CalcMolFormula(mol)  # 'C17H30'
   ```

3. **Carga las picos en la GUI:**
   - Las columnas `crosspeaks` y `c13_shifts` provienen de Fase 1b (blob-detection + traceback).
   - La multiplicidad de cada crosspeak está codificada en los datos de imagen (DEPT phase).
   - En la tabla de la GUI, rellena cada fila con:
     - `delta_c` del array `c13_shifts`
     - `delta_h` del array `crosspeaks[i]` (segunda coordenada)
     - `mult`: inferido del tipo de crosspeak (típicamente CH3, CH2, CH vienen con crosspeak; Cq no).
     - `clase`: **vacío** (modo predicción).

4. **Compara la predicción:**
   - El vector predicho (candidato 0) debe coincidir o estar muy cerca de `y_pred_assisted_v2` del parquet.
   - Pequeñas discrepancias (~±1 en alguna clase) son normales si el modelo fue reentrenado o τ es diferente.

**Nota:** Fase 1b (multiplicidad) está fuera del alcance de esta app. Los datos de `crosspeaks` y `c13_shifts` se extraen externamente. Para workflows completamente automatizados, ver Exp G (generador de estructuras).

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
