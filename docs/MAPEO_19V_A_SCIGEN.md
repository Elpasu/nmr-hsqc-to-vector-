# Mapeo del vector de 19 clases a las 13 categorías de Sci-Gen

**Fecha:** 2026-08-14
**Fuentes leídas:**
- 19 clases: `experiments/H_inferencia_experimental/smiles_classifier.py:45-91`
  (`classify_carbon`, puerto fiel de `Gen_vector.py`) + orden de `config/db.yaml:36-55`.
- 13 categorías: `Sci-Gen/src/RESmaygen.cpp:345` (`NMR_CATEGORIES = 13`) y
  `classifyCarbonNMR` (mismo archivo, ~línea 380).

**Para qué:** Sci-Gen poda la generación comparando un vector de conteos de carbono contra
el objetivo (`--nmr-vector`). Este documento fija cómo se traduce nuestro vector de 19 al
suyo de 13, y qué se pierde en el camino.

---

## 0. Por qué esto recién ahora es posible

Las 13 categorías de Sci-Gen se cuentan así (`RESmaygen.cpp:595-598`):

```cpp
int computed[NMR_CATEGORIES] = {};
for (int i = 0; i < n; i++) {
    int cat = classifyCarbonNMR(a, n, syms, i, isAromatic);
    if (cat >= 0 && cat < NMR_CATEGORIES) computed[cat]++;
}
```

Recorre **todos los átomos** y cuenta **cada carbono**. No hay deduplicación por simetría:
`Σ computed == nC`. Es decir, el vector que Sci-Gen calcula internamente **siempre fue un
vector de carbonos totales**.

Nuestro vector de señales (benceno = 1 en `=CH/Ar`) nunca fue comparable con eso. El vector
de Exp J (benceno = 6) sí lo es. Esa es la razón por la que este mapeo se puede escribir
ahora y antes no.

---

## 1. Las 13 categorías de Sci-Gen

Derivadas de la cascada de decisión de `classifyCarbonNMR`:

| # | Categoría | Condición |
|---|---|---|
| 0 | CH3 sp3 normal | no sp2, sin vecino O/N, nH=3 |
| 1 | CH2 sp3 normal | ídem, nH=2 |
| 2 | CH sp3 normal | ídem, nH=1 |
| 3 | Cq sp3 normal | ídem, nH=0 |
| 4 | CH3 sp3 hetero | no sp2, **algún vecino O o N**, nH=3 |
| 5 | CH2 sp3 hetero | ídem, nH=2 |
| 6 | CH sp3 hetero | ídem, nH=1 |
| 7 | Cq sp3 hetero | ídem, nH=0 |
| 8 | CH2 sp2 normal | sp2, no carbonilo, nH=2 |
| 9 | CH sp2 normal | sp2, no carbonilo, nH=1 |
| 10 | Cq sp2 normal | sp2, no carbonilo, nH=0 |
| 11 | CH sp2 carbonilo | sp2, doble enlace a O o N, nH=1 |
| 12 | Cq sp2 carbonilo | sp2, doble enlace a O o N, nH=0 |

**`hetero` en Sci-Gen es O **o** N en un solo bucket** — no los distingue. Y **no incluye
halógenos ni S**: `RESmaygen.cpp` solo chequea `"O" | "N" | "[N+]" | "[O-]"`.

---

## 2. La tabla de mapeo

| # | Clase 19v | → Sci-Gen | Tipo |
|---|---|---|---|
| 0 | CH3 | 0 | exacto |
| 1 | CH2 | 1 | exacto |
| 2 | CH | 2 | exacto |
| 3 | Cq | 3 | exacto |
| 4 | CH3-O | 4 | **fusión** con 8 |
| 5 | CH2-O | 5 | **fusión** con 9 |
| 6 | CH-O | 6 | **fusión** con 10 |
| 7 | Cq-O | 7 | **fusión** con 11 |
| 8 | CH3-N | 4 | **fusión** con 4 |
| 9 | CH2-N | 5 | **fusión** con 5 |
| 10 | CH-N | 6 | **fusión** con 6 |
| 11 | Cq-N | 7 | **fusión** con 7 |
| 12 | =CH2 | 8 | exacto |
| 13 | =CH/Ar | 9 | exacto |
| 14 | Cqsp2 | 10 **o** 12 | **división** — indeterminado |
| 15 | Aldeh | 11 | **fusión** con 16 |
| 16 | Imina | 11 | **fusión** con 15 |
| 17 | C-2X | 4/5/6/7 | **indeterminado** (falta el nH) |
| 18 | C-3X | 4/5/6/7 | **indeterminado** (falta el nH) |

### Los cuatro puntos de fricción

**(a) O y N se fusionan.** Sci-Gen tiene un solo bucket `hetero`. Nuestras 8 clases
`{CH3,CH2,CH,Cq}×{O,N}` caen en sus 4 categorías 4-7. La fusión es una **suma exacta**
(`cat4 = v[4] + v[8]`, etc.), así que no pierde poder de poda sobre el conteo total —
solo pierde la distinción química O vs N, que Sci-Gen nunca tuvo.

**(b) Aldeh + Imina → 11.** Otra suma exacta: `cat11 = v[15] + v[16]`.

**(c) `Cqsp2` se DIVIDE, y esta es la que duele.** Nuestra clase 14 mete en la misma bolsa
el Cq aromático/olefínico y el carbonilo cuaternario (cetona, éster, amida). Sci-Gen los
separa en 10 y 12. Es el único lugar donde **Sci-Gen es más fino que nosotros**, y no se
puede resolver sumando: dado `v[14] = 3` no sabemos si es `(3,0)`, `(2,1)`, `(1,2)` o `(0,3)`.

**(d) C-2X / C-3X no llevan el nH.** Nuestras clases 17 y 18 dicen "carbono sp3 con 2 (o 3+)
vecinos heteroátomo" pero **no cuántos hidrógenos tiene**. Sci-Gen necesita el nH para elegir
entre 4, 5, 6 y 7. No se puede reconstruir.

> Ojo con el nombre: en `classify_carbon`, `nX` cuenta **cualquier vecino pesado no-carbono**,
> no solo halógenos. En un dataset CHON puro, `C-2X`/`C-3X` son acetales, cetales, aminales y
> ortoésteres — no compuestos halogenados. La "X" del nombre es engañosa.

---

## 3. Cómo se traduce en la práctica: el comodín `-1`

Sci-Gen ya soporta categorías sin restringir. En `RESmaygen.cpp:348`:

```cpp
int nmrVector[NMR_CATEGORIES] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};
```

y el chequeo es `if (nmrVector[c] >= 0 && computed[c] != nmrVector[c])`. **Un `-1` significa
"no chequear esta categoría".** Eso resuelve (c) y (d) sin tocar el C++: lo que no se puede
determinar se manda como `-1` y se sigue podando con el resto.

Traducción propuesta, dado `v` = vector de 19 de Exp J:

| Sci-Gen | Valor | Cuándo es exacto |
|---|---|---|
| 0 | `v[0]` | siempre (dataset CHON) |
| 1 | `v[1]` | siempre |
| 2 | `v[2]` | siempre |
| 3 | `v[3]` | siempre |
| 4 | `v[4]+v[8]` | **solo si** `v[17]==0 and v[18]==0`, si no `-1` |
| 5 | `v[5]+v[9]` | ídem |
| 6 | `v[6]+v[10]` | ídem |
| 7 | `v[7]+v[11]` | ídem |
| 8 | `v[12]` | siempre |
| 9 | `v[13]` | siempre |
| 10 | `-1` | nunca (fusionado con 12 en nuestra clase 14) |
| 11 | `v[15]+v[16]` | siempre |
| 12 | `-1` | nunca |

**Resultado: 7 de 13 categorías siempre restringidas** (0,1,2,3,8,9,11), 4 más
(4,5,6,7) cuando la molécula no tiene C-2X/C-3X, y 2 nunca (10,12).

### Nota sobre la restricción de suma

`Σ v == nC` (invariante de Exp J) es información fuerte que este esquema **desaprovecha**:
Sci-Gen no tiene un constraint de suma, solo el chequeo por categoría. Con 10 y 12 en `-1`,
la suma de las categorías restringidas queda por debajo de nC y no hay forma de exigir que
el resto complete. Recuperar eso sí requeriría un cambio aditivo en el C++.

---

---

## 4. Cuánto pesa cada agujero — medido

Sobre las 14 428 moléculas del val congelado (`predictions_nmr_202k_j_carbonos_deg.parquet`):

| Situación | Moléculas | % |
|---|---|---|
| Tienen `Cqsp2` (`v[14] > 0`) → pierden cat 10 y 12 | 13 114 | **90,89 %** |
| Tienen `C-2X`/`C-3X` → pierden además cat 4-7 | 1 038 | 7,19 % |
| Ambos | 796 | 5,52 % |
| Ninguno → mapeo 11 de 13 exacto | 1 072 | 7,43 % |

**Carbonos sin restringir: 3,16 por molécula en promedio, el 27,6 % del total.**

Invariante verificada de paso: `sum(y_true) == nC` en **14 428 / 14 428 (100,00 %)**, contra
el conteo de carbonos de RDKit sobre el SMILES. El vector de Exp J es carbonos totales de
verdad. (Su promedio en el val congelado es 11,47 C/molécula, algo menor que el 13,3 del
spec, que se midió sobre el dataset completo — no afecta la invariante.)

### Lectura

El problema **(d)** es menor: en el 93 % de las moléculas las categorías 4-7 se restringen
exactamente. El problema **(c) es el grande**: el 91 % de las moléculas tiene al menos un
`Cqsp2`, así que en la enorme mayoría de los casos las categorías 10 y 12 van en `-1` y se
pierde algo más de un cuarto de los carbonos como restricción.

### La salida para (c), sin reentrenar

No se puede dividir `v[14]` en 10 y 12, pero **sí se conoce su suma**: `cat10 + cat12 = v[14]`,
exactamente. Sci-Gen hoy no tiene constraint de suma sobre un grupo de categorías, solo
igualdad por categoría. Agregarlo es un cambio **aditivo** en el C++ (compatible con la regla
dura 2 de `Sci-Gen/CLAUDE.md`: archivos nuevos, gateado por flag) y recuperaría la mayor
parte de ese 27,6 %. Es el candidato número uno de la fase de integración.

Lo mismo aplica a la restricción global `Σ v == nC`, que este esquema hoy desaprovecha por
completo (ver §3).

---

## 5. Pendientes concretos, en orden

1. **Implementar el constraint de suma por grupo en Sci-Gen** (`cat10 + cat12 == v[14]`, y
   opcionalmente `Σ cat == nC`). Aditivo, gateado por flag. Es lo que más poda recupera.
2. **Verificar la monotonía de la poda parcial.** `checkNMRVectorSp3`
   (`RESmaygen.cpp:454-468`) poda solo por exceso (`computed[c] > nmrVector[c]`). Con el
   vector de carbonos ese criterio deja de podar moléculas válidas por colapso de simetría
   —que era el defecto del vector de señales— pero falta confirmar que la categoría de un
   carbono no puede *bajar* durante la construcción (un carbono con un enlace parece CH3 y
   pasa a CH2 al agregar el siguiente). **No asumir resuelto.**
3. **Decidir qué hacer con los K vectores.** El modelo emite 1,37 vectores promedio a τ=1,0
   (Exp J, cobertura 94,17 %). Sci-Gen toma **un** `--nmr-vector`. Hay que decidir si se
   corre K veces y se unen los resultados, o si se relaja el vector a la intersección.

## 5. Lo que NO hace falta arreglar

El bug de aromaticidad de `refinarClases` (tolueno: 7 clases en vez de 5), marcado como
condición de entrada en `Sci-Gen/docs/FACTIBILIDAD_19V.md` §5, **no está en el camino
crítico**. `refinarClases` solo se llama desde `Sci-Gen/tests/cpp/test_symmetry_main.cpp`;
no está cableado al generador. Y con el vector de carbonos totales no hay que colapsar por
simetría, así que no hace falta cablearlo. El bug sigue existiendo — deja de bloquear.
