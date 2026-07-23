# Diseño — Oráculo v2 (restricciones de heteroátomos) + auditoría del post-procesamiento

**Fecha:** 2026-07-23
**Autor:** Lucas Passaglia (con Claude Code)
**Estado:** propuesto (pendiente de revisión)

---

## 1. Contexto y motivación

La EMA **asistida** del mejor modelo (Exp E Fase 3, Set Transformer) está en **91.35%** y no
se mueve por las palancas ya probadas:

- **Exp F (cabeza Poisson + 250 épocas):** no mejoró.
- **Estudio de escalado (10/25/50/75/100%):** meseta — más datos no mueve la aguja.

Con la arquitectura y la loss agotadas como palancas, el margen que queda para la EMA
**asistida** está en el **post-procesamiento** (el oráculo). Este documento audita el oráculo
actual y propone una mejora (v2) puramente de post-procesamiento sobre el **mismo checkpoint**
(sin reentrenar), directamente comparable con el 91.35%.

---

## 2. Auditoría del oráculo actual

### 2.1 El condicionante tiene 8 valores; el oráculo usa 2

`dataset_e3.py:67` arma el condicionante (`cond_dim: 8`, ver `config/db.yaml`):

| idx | valor | ¿lo usa el oráculo? |
|-----|-------|---------------------|
| 0 | total de carbonos (= suma del vector) | ✅ `sum(pred) == total` |
| 1 | total CH2-type (CH2+CH2-O+CH2-N+=CH2) | ✅ `sum(pred[IDX_CH2]) == ch2` |
| 2 | C (fórmula) | ❌ (redundante con idx 0) |
| 3 | H | ❌ |
| 4 | **N** | ❌ |
| 5 | **O** | ❌ |
| 6 | S | ❌ (no se usa; dataset CHON) |
| 7 | Hal | ❌ (no se usa; dataset CHON) |

`ajustar_conteo_doble_exacto` (`evaluate.py:58`) fuerza dos igualdades exactas: total de
carbonos y total de CH2-type. **Respuesta a la pregunta original:** sí, si se pasa la FM, el
vector asistido siempre cierra en el total de carbonos correcto (`sum(pred)==C`), por
construcción. Pero se usan **2 de las 8** restricciones que la FM regala.

### 2.2 Causa raíz de la confusión CH2 ↔ CH2-N

`IDX_CH2 = [1, 5, 9, 12]` = {CH2, CH2-O, CH2-N, =CH2} forman **un solo cupo**. El oráculo
fuerza que la *suma* de los cuatro sea `ch2_real`, pero **no puede distinguir CH2 de CH2-N
dentro del cupo**. Si el modelo intercambia CH2↔CH2-N, la suma no cambia → el error sobrevive
a la corrección. Esto es una limitación **estructural** del oráculo actual, observada en la GUI.

### 2.3 Ranking de palancas disponibles

- **N/O ausentes (exacto, fuerte):** si la FM dice 0 de un elemento, todas las clases que lo
  requieren son 0. Airtight. Es la base de v2.
- **Cotas superiores `≤3·N`, `≤2·O` (débil, riesgosa):** un N puede unir 0–3 carbonos; la cota
  rara vez es ajustada y puede romper predicciones correctas. **Fuera de alcance.**
- **Balance de H (no ayuda a CH2/CH2-N):** CH2 y CH2-N ambos aportan 2 H; el balance de H no
  los distingue. **Fuera de alcance.**

---

## 3. Definiciones exactas de las clases (fuente: `Gen_vector.py`)

`classify_carbon` (`E:\Proyectos\SciTrix\MYGEN\HSQC_a_Vector\Gen_vector.py:55`) define
`nX` = nº de vecinos pesados que **no son carbono** (heteroátomos; con dataset CHON = N u O):

- **sp3, `nX == 2` → C-2X** (idx 17); **`nX >= 3` → C-3X** (idx 18). *(acetales, aminales,
  ortoésteres — X = heteroátomo, NO halógeno).*
- **sp3, `nX <= 1`:** si tiene vecino O → `CH*-O` (idx 4–7); elif tiene vecino N → `CH*-N`
  (idx 8–11); else → `CH*_normal` (idx 0–3). *(`has_O` tiene prioridad sobre `has_N`).*
- **sp2 con H, doble enlace a O → Aldeh** (idx 15); **doble enlace a N → Imina** (idx 16).
- **sp2 cuaternario (nH==0) → Cqsp2** (idx 14) — incluye aromáticos y **C=O de cetona/éster**.
  Por eso **Cqsp2 NO requiere O** y no se toca.

---

## 4. Oráculo v2 — reglas airtight

Condiciones **necesarias** derivadas de §3 (si la condición de la FM se cumple, la clase DEBE
ser 0; forzarla a 0 no puede introducir error):

| Regla (FM) | Clases forzadas a 0 | Índices |
|---|---|---|
| **N == 0** | CH3-N, CH2-N, CH-N, Cq-N, Imina | 8, 9, 10, 11, 16 |
| **O == 0** | CH3-O, CH2-O, CH-O, Cq-O, Aldeh | 4, 5, 6, 7, 15 |
| **N + O < 2** | C-2X | 17 |
| **N + O < 3** | C-3X | 18 |

(Con N==0 y O==0, las dos últimas se activan solas: N+O=0 < 2 y < 3.)

### 4.1 Algoritmo

```
def ajustar_conteo_hetero(pred_raw, total_real, ch2_real, n_atoms, o_atoms):
    pred = pred_raw.copy()
    # 1) Zeroing por ausencia (airtight)
    if n_atoms == 0:     pred[[8, 9, 10, 11, 16]] = 0.0
    if o_atoms == 0:     pred[[4, 5, 6, 7, 15]]    = 0.0
    if n_atoms + o_atoms < 2:  pred[17] = 0.0
    if n_atoms + o_atoms < 3:  pred[18] = 0.0
    # 2) Ajuste de doble restricción existente (total + cupo CH2), sin cambios
    return ajustar_conteo_doble_exacto(pred, total_real, ch2_real)
```

Al poner `pred_raw = 0.0` en las clases prohibidas **antes** del ajuste de doble restricción:
`floor → 0` y `resto → 0`, por lo que esas clases nunca se incrementan (resto más bajo) ni se
decrementan (valor 0). La masa liberada se reparte automáticamente entre las clases permitidas
según resto decimal. En el cupo CH2, si una molécula no tiene N, `ch2_real` se cubre con
{CH2, CH2-O (si O>0), =CH2} → empuja hacia **CH2 pleno**. Fuera del cupo, C-2X/C-3X salen del
pool de "resto".

### 4.2 Propiedades

- **Seguridad:** v2 solo modifica clases cuyo conteo verdadero es 0 (elemento ausente). No
  puede empeorar esas clases. El único riesgo (menor, medible) es que la redistribución empuje
  masa a otra clase equivocada; las restricciones total + cupo CH2 lo acotan.
- **Superconjunto de v1:** en moléculas con N>0 y O>0 y N+O≥3, v2 ≡ v1 (no dispara ninguna
  regla). Solo actúa donde hay una ausencia demostrable.
- **Capacidad garantizada:** CH2, CH, CH3, Cq, Cqsp2, =CH/Ar nunca se anulan (no requieren
  heteroátomo), así que el cupo CH2 y el pool de resto siempre pueden satisfacer las igualdades.
- **Comparabilidad:** mismo checkpoint, mismo val congelado, misma seed. La diferencia de EMA
  aísla exactamente el cambio de post-procesamiento.

---

## 5. Alcance de la implementación

### 5.1 Dentro de alcance

1. **`evaluate.py` (E3):** nueva función `ajustar_conteo_hetero` + modo `--oraculo all` que
   corre y compara **cruda / v1 (doble) / v2 (hetero)** en una tabla de 3 columnas
   (EMA global + por entorno) y muestra el mapa de confusiones de v2. Los modos existentes
   (`on`/`off`/`both`) quedan intactos.
2. **Test:** `tests/test_oraculo_hetero.py` (numpy puro, sin torch) que verifica:
   - zeroing correcto por cada regla (N==0, O==0, N+O<2, N+O<3);
   - que las igualdades total y cupo CH2 se siguen cumpliendo tras el zeroing;
   - que con N>0, O>0, N+O≥3 el resultado es idéntico a `ajustar_conteo_doble_exacto`;
   - el caso concreto CH2-N→CH2 en una molécula sin N.
3. **`dump_predictions.py` (E3):** columna adicional `y_pred_assisted_v2` (para verla en la GUI
   junto a la asistida v1).
4. **`gui_inspector.py`:** en el selector de modo, agregar "Asistida v2 (hetero)" si la columna
   existe (compatible con parquets viejos).

### 5.2 Fuera de alcance (documentado como futuro)

- Cotas superiores `≤3·N`, `≤2·O` para el caso N>0/O>0 (flojas, riesgosas).
- Balance de H (no distingue CH2 de CH2-N).
- Formulación como programa entero conjunto (ILP) con todas las restricciones.
- Reentrenar el modelo (esto es solo post-procesamiento).

---

## 6. Criterios de éxito / fracaso

- **Éxito:** EMA asistida v2 > v1 (91.35%) en el val congelado, con caída medible del conteo de
  confusiones CH2↔CH2-N (y CH*-O/CH*-N en general) en el mapa de confusiones.
- **Neutro:** v2 ≈ v1 → la confusión está dominada por moléculas **con** N (donde v2 no
  dispara); indicaría que el próximo paso es la cota `≤3·N` o un modelo que distinga mejor por
  shift. El resultado igual es informativo.
- **Fracaso (improbable por construcción):** v2 < v1 → la redistribución rompe más de lo que
  arregla; revisar el reparto por resto.

---

## 7. Verificación

- Test de oráculo ejecutable localmente (numpy puro, sin torch/cluster).
- `evaluate.py --oraculo all` en el cluster sobre el checkpoint Fase 3 existente (un job de
  ~30 min, sin reentrenar).
- Inspección visual en la GUI: filtrar por confusión CH2/CH2-N y comparar v1 vs v2.
