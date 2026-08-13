# Exp J — decisiones de diseño

## Por qué una carpeta nueva y no una variante dentro del E3

El Exp I (sweep de hiperparámetros) vive **dentro** de `E3_dos_conjuntos/` justamente porque su
validez depende de usar el `train.py` bit-a-bit idéntico al del checkpoint congelado.

Exp J es el caso opuesto: los labels, el dataset y la semántica del condicionante divergen de
verdad. Copiar es lo correcto acá — es la convención estándar del proyecto (carpeta autocontenida)
y evita que un cambio pensado para J rompa el E3 en producción.

## Por qué degeneración y no la integral cruda

El químico carga **H** (2, 3, 6…), que es lo que lee del ¹H integrado. El pipeline divide por los
H-por-carbono —que ya conoce de la multiplicidad— y le pasa al modelo el **número de carbonos
equivalentes**.

Hacer la división afuera es inyectar conocimiento químico explícito como condicionante, que es la
palanca que más rindió en toda la serie histórica (CH2, Fórmula Molecular). Una división no es una
operación natural para una red; no hay razón para hacérsela aprender.

## Por qué los cuaternarios no llevan degeneración

En un ¹³C real las intensidades no son cuantitativas (NOE, tiempos de relajación distintos). Darle
al modelo la degeneración de un carbono sin H sería entrenarlo con información que no va a existir
en el espectro experimental — el modelo aprendería a depender de algo que en inferencia no está.

Lo único que acota los cuaternarios es la suma total de la FM, y eso es exactamente lo que produce
el 2,3 % de ambigüedad residual del techo.

## Por qué un solo archivo de modelo con `peak_features` configurable

Dos archivos casi idénticos se desincronizan sin avisar. Si J-A y J-0 corrieran sobre modelos
distintos que divergieron en algo, la comparación entre ambas dejaría de medir el aporte de la
integración y pasaría a medir esa divergencia — sin que nada tire error.

Por el mismo motivo los dos configs leen el **mismo** `.npz`: el control recorta la columna en el
dataset, no usa otro archivo.

## Por qué el gate de verificación del clasificador

`make_labels_totales.py` regenera los labels **viejos** y los compara contra
`vectors_13c_19v_202465.npy` antes de escribir nada. Si no coinciden al 100 %, aborta.

Sin ese gate, un clasificador sutilmente distinto produciría un ground truth nuevo corrupto sin
tirar ningún error — el modo de falla exacto que la regla dura 7 existe para prevenir. Verificado:
100,0000 % sobre las 202 465 moléculas del dataset completo (corrida real de
`make_labels_totales.py`), 0 discrepancias.

## Por qué `oraculo.py` se copia sin cambios

La lógica de forzar `sum(pred) == total` y el cupo de CH2 es idéntica. Lo único que cambia es que
`total` ahora vale C de la fórmula en vez del número de señales — un cambio de datos, no de
algoritmo.

De hecho el esquema nuevo es **más robusto**: antes `total` se contaba del espectro y dos señales
solapadas lo arruinaban; ahora sale de la FM, exacto.
