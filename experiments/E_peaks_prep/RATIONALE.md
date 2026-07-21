# Exp E — Fase 1: Extracción y Validación de Picos — Rationale

## Por qué

Exp C (GAP) mejoró la EMA cruda levemente pero las confusiones de clase
(`Cqsp2`↔`=CH/Ar`, `CH2`↔`CH2-N`) persisten idénticas en V10, Exp B y Exp C —
tres arquitecturas distintas. Evidencia de que el cuello de botella es de
representación, no arquitectónico. La auditoría de pipeline
(`scripts/audit_data_pipeline.py`) mostró que la imagen HSQC es 99.2% espacio
vacío. Esta fase reemplaza la imagen por una lista compacta de picos reales.

## Calibración

Encontrada en `E:\Proyectos\SciTrix\ScitrixDB\DB-Batch0\Genera mapas de pkl v2.py`
(script original del dataset, fuera de este repo, solo consultado como
referencia — no se ejecuta ni se modifica):

- δC: `[0, 220]` ppm, binning uniforme, 256 bins.
- δH: `[-1, 15]` ppm, binning uniforme, 256 bins.
- Canal 0 (fila=δC, columna=δH) = gaussiana DEPT, sigma=0.5, escalada por N_H
  (CH2 negativo, CH/CH3 positivo).
- Canal 1 = tipo de carbono normalizado (CH=0.33, CH2=0.67, CH3=1.0).
- Los H de un mismo carbono se pintan en el mismo pixel (no es colisión real).
  La colisión real es cuando dos carbonos DISTINTOS caen tan cerca que sus
  gaussianas se funden en un blob — eso es lo que valida esta fase.

## Alcance

Solo extracción + validación. No hay modelo de conjuntos todavía (Fase 2,
spec separado, depende de qué tan limpia salga esta extracción).

Ver el spec completo: `docs/superpowers/specs/2026-07-21-exp-e-fase1-extraccion-picos-design.md`.
