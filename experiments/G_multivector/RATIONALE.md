# Exp G — Rationale

**Objetivo:** cobertura@K, no EMA. El vector alimenta un generador de estructuras;
su trabajo es achicar el espacio de generacion sin perder el vector verdadero.
Perder especificidad (emitir 2-3, generar de mas) es tolerable; perder al verdadero
es gravisimo.

**Hipotesis:** como el 85% de las fallas conserva la multiplicidad (nH) — el modelo
acierta cuantos H tiene el carbono y solo confunde el entorno dentro de ese nH —,
un set chico de candidatos generados moviendo conteos intra-nH cubre al verdadero.

**Evidencia (parquet Fase 3, val congelado 14428):** top-1 91.36% -> +1 swap intra-nH
98.18% -> cap 98.72%. 984 moleculas a exactamente 1 swap; 185 (1.28%) necesitan
cross-nH (fuera de alcance v1).

**Exito:** cobertura@K >= 98% con K promedio <= 3, top-1 == oraculo v2.
**Fracaso:** cobertura muy por debajo de la curva teorica a K chico -> revisar el
generador o el ranking.

Spec: docs/superpowers/specs/2026-07-24-exp-g-multivector-coverage-design.md

## Fase 1b

Fase 1 emitia K vectores SIEMPRE (K prom == K), sin especificidad. Fase 1b emite
alternativas solo donde hay duda real (costo extra de L1 < tau), dando K adaptativo:
molecula segura -> 1 vector; con dudas -> pocas mas. Objetivo: misma cobertura que
Fase 1 (~97-98%) con K promedio mucho mas bajo. tau y K_max se calibran con el barrido.
Spec: docs/superpowers/specs/2026-07-24-exp-g-fase1b-guiada-incertidumbre-design.md
