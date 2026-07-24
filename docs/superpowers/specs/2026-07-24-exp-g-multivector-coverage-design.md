# Diseño — Exp G: multi-vector (cobertura@K) para alimentar el generador de estructuras

**Fecha:** 2026-07-24
**Autor:** Lucas Passaglia (con Claude Code)
**Estado:** propuesto (pendiente de revisión)

---

## 1. Cambio de objetivo (leer primero)

El vector **no es el resultado final**: alimenta un **generador de estructuras** aguas abajo.
Su función es **achicar el espacio de generación sin perder la estructura correcta**. Por eso la
métrica deja de ser EMA top-1 y pasa a ser **cobertura@K**: que el vector verdadero esté *sí o sí*
dentro de los K vectores que emitimos. Perder especificidad (emitir 2-3 y generar de más) es
tolerable; **perder al verdadero es gravísimo** (la molécula correcta queda fuera de la generación).
Qué estructura generada es la correcta se resuelve en otra línea de trabajo, aguas abajo.

Motivación completa en `docs/Runs/RESULTS.md`, sección "► PRÓXIMO OBJETIVO — Cobertura@K", y en
la memoria `coverage-at-k-objetivo-exp-g`.

## 2. Evidencia (medida sobre el parquet Fase 3 Set Transformer, val congelado 14428)

El 85% de las fallas conserva la multiplicidad (nH): el modelo acierta cuántos H tiene el carbono y
solo confunde el *entorno* dentro de ese grupo de nH (CH2↔CH2-N, CH↔CH-N, Imina↔=CH/Ar). Cobertura
si el generador de candidatos mueve conteos **dentro del mismo grupo de nH**, manteniendo las
restricciones de la FM, **sin reentrenar**:

| Candidatos (K) | Cobertura (verdadero ∈ set) |
|---|---|
| top-1 | 91.36% |
| + 1 swap intra-nH | **98.18%** |
| + 2 swaps | 98.70% |
| cap intra-nH | 98.72% |

984 moléculas (6.82%) están a **exactamente 1 swap**. Solo **185 (1.28%)** necesitan un movimiento
**cross-nH** (multiplicidad mal) — ese es el límite duro dato/modelo, fuera del alcance de v1.

## 3. Grupos de multiplicidad (nH) — el eje de la ambigüedad

Derivados del clasificador real (`Gen_vector.py`): cada clase tiene un nH fijo. Los grupos son:

| nH | clases (índices) |
|----|------------------|
| 3H | CH3 (0), CH3-O (4), CH3-N (8) |
| 2H | CH2 (1), CH2-O (5), CH2-N (9), =CH2 (12) |
| 1H | CH (2), CH-O (6), CH-N (10), =CH/Ar (13), Aldeh (15), Imina (16) |
| 0H | Cq (3), Cq-O (7), Cq-N (11), Cqsp2 (14), C-2X (17), C-3X (18) |

El grupo 2H coincide con `IDX_CH2` del oráculo (ya restringido por la FM).

## 4. Diseño de la solución — generador de candidatos post-hoc (sin reentrenar)

### 4.1 Insumo: output crudo del modelo

Hoy `dump_predictions.py` guarda el redondeo (crude/v1/v2) pero **no** los conteos reales (floats)
que salen del modelo. La incertidumbre vive ahí: si en el grupo 2H el modelo dice
`CH2=1.4, CH2-N=0.6` y la FM fija 2 CH2-type, los dos candidatos son `(2,0)` y `(1,1)`, rankeados
por esa masa. **Cambio requerido:** agregar la columna `y_pred_raw` (19 floats) al dump. Es el único
cambio que toca el cluster/torch; una corrida de dump sobre el mejor checkpoint.

### 4.2 Generador de candidatos (numpy puro, corre local sin GPU)

Dado `r` (raw floats, 19) y la FM (total de señales, cupo CH2, N, O):

1. **Ancla FM-consistente:** el candidato base es el oráculo v2 (`ajustar_conteo_hetero`), que ya
   satisface total + cupo CH2 + zeroing por ausencia. Reusa `oraculo.py`.
2. **Total por grupo de nH:** se fija al del ancla (= redondeo del modelo). v1 de Exp G **no mueve
   conteos entre grupos** (respeta la multiplicidad, que el modelo acierta el 85% de las veces).
   Esto acota la cobertura a ~98.7% (cross-nH = fase 2).
3. **Enumeración intra-grupo:** para cada grupo, enumerar las reparticiones enteras del total del
   grupo entre sus clases *permitidas* (excluye las anuladas por ausencia de heteroátomo), a
   distancia L1 pequeña de `r` restringido al grupo. En la práctica solo ramifica el grupo con una
   fracción ambigua (≈0.5); los demás tienen una sola opción.
4. **Combinación y ranking:** producto cartesiano de las alternativas por grupo, filtrando por las
   restricciones globales de la FM (total, cupo CH2). Rankear cada vector candidato por
   `sum |c - r|` (menor = más probable) y emitir los **top-K**.
5. El candidato top-1 debe coincidir con el oráculo v2 (sanity check).

### 4.3 Métrica: curva cobertura@K

`coverage@K = fracción de moléculas del val congelado con y_true dentro de los K candidatos`.
Deliverable: la curva K=1..5 (o hasta saturar), para elegir el K operativo (~2-3) que da ~100%
alcanzable (techo 98.7% en v1). Reportar también el **K promedio y máximo** emitido (costo de
generación aguas abajo) y la distribución de cuántos candidatos se emiten por molécula.

## 5. Estructura de la implementación

Carpeta nueva **`experiments/G_multivector/`** (autocontenida; reusa `oraculo.py` de E3 vía copia,
según convención del proyecto — no importar de otro experimento):

- `candidates.py` (numpy puro): `generate_candidates(raw, total, ch2, n_atoms, o_atoms, K) -> list[np.ndarray]`.
- `coverage.py`: carga el parquet con `y_pred_raw`, corre el generador, computa la curva cobertura@K.
- `tests/test_candidates.py` (numpy puro, **corre local sin torch**): top-1 == oráculo v2; todos los
  candidatos son FM-consistentes (suma + cupo CH2 + zeroing); el caso 2H `(1.4,0.6)`→`{(2,0),(1,1)}`;
  cobertura monótona en K; no duplica candidatos.
- `oraculo.py`: copia de `experiments/E3_dos_conjuntos/oraculo.py` (fuente de las reglas v2).
- `README.md`, `RATIONALE.md`.

Cambio en `experiments/E3_dos_conjuntos/dump_predictions.py`: agregar columna `y_pred_raw`.

## 6. Checkpoint y datos

Usar el **mejor checkpoint actual** (Set Transformer Fase 3 en Intel XPU/Clementina, val loss 0.0086,
EMA 92.14% v2). La evidencia de la §2 se midió sobre el parquet del A10 (91.36%); la estructura de
confusiones es la misma, así que la cobertura esperada es equivalente (a re-medir con el dump nuevo).
Val congelado idéntico (14428) — comparabilidad con todo el proyecto (rule 8).

## 7. Alcance

**Dentro:** columna `y_pred_raw` en el dump; generador intra-nH; curva cobertura@K; tests locales.

**Fuera (fase 2, documentado):**
- Candidatos **cross-nH** (mover ±1 entre grupos) para el 1.28% de multiplicidad mal → subir el
  techo por encima de 98.7%.
- **Reentrenar** el modelo para emitir una distribución calibrada por grupo de nH (mejor ranking ⇒
  igual cobertura con K más chico). Solo si el K operativo resulta demasiado grande.
- Integración con el generador de estructuras (otra línea de trabajo).

## 8. Criterios de éxito / fracaso

- **Éxito:** cobertura@K ≥ 98% con K promedio ≤ 3 en el val congelado, con top-1 == oráculo v2.
- **Neutro/informativo:** si K necesario para 98% es grande (>5), indica que el ranking por masa
  blanda no discrimina bien → justifica la fase 2 (reentrenar con distribución calibrada).
- **Fracaso:** cobertura@K se queda muy por debajo de la curva teórica (§2) a K chico → revisar el
  generador (probablemente no está explorando el swap correcto o el ranking está invertido).

## 9. Verificación

- `tests/test_candidates.py` ejecutable local (numpy puro, sin torch/GPU).
- `coverage.py` corre 100% local una vez dumpeado `y_pred_raw` → iteración de tuning sin cola de GPU.
- Sanity: `coverage@1` debe reproducir la EMA asistida v2 del checkpoint (~92.1%).
