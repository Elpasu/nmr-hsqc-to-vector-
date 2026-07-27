# coding: utf-8
"""Exp H -- interfaz Streamlit local para predecir el vector desde picos
experimentales + FM. Corre en TU PC (no en el cluster).

Requisitos:  pip install streamlit torch pandas pyyaml numpy
Uso:         streamlit run experiments/H_inferencia_experimental/app_inferencia.py
Ajusta CHECKPOINT_PATH a la ruta local del checkpoint (scp desde Clementina).
"""
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import adapter  # noqa: E402
import predict_core  # noqa: E402

# Ruta al checkpoint local (scp desde Clementina). Portable: hardcodeada si
# existe, si no relativa al repo.
_HARDCODED = r"E:\Proyectos\SciTrix\nmr-hsqc-to-vector\checkpoints_local\nmr_202k_e3_settransformer_2sets_19v_best.pth"
_REL = os.path.join(_REPO, "checkpoints_local",
                    "nmr_202k_e3_settransformer_2sets_19v_best.pth")
CHECKPOINT_PATH = _HARDCODED if os.path.exists(_HARDCODED) else _REL

st.set_page_config(page_title="NMR Inferencia experimental", layout="wide")


@st.cache_resource
def _load():
    class_names, norm = adapter.load_configs(_REPO)
    cfg = yaml.safe_load(open(
        os.path.join(_REPO, "experiments", "E3_dos_conjuntos",
                     "config_settransformer.yaml"), encoding="utf-8"))
    model = predict_core.load_model(CHECKPOINT_PATH, cfg["model"])
    return class_names, norm, model


st.title("NMR HSQC -> vector: inferencia sobre datos experimentales")

if not os.path.exists(CHECKPOINT_PATH):
    st.error(f"No encuentro el checkpoint en:\n{CHECKPOINT_PATH}\n"
             "Copialo desde Clementina (ver README).")
    st.stop()

class_names, norm, model = _load()

col1, col2 = st.columns([1, 1])
with col1:
    formula = st.text_input("Fórmula molecular (ej. C10H12N2O)", value="C2H6O")
with col2:
    tau = st.slider("τ (Fase 1b)", 0.0, 3.0, 1.5, 0.25)
    k_max = st.slider("K_max", 1, 10, 6, 1)

st.caption("Una fila por carbono. δH vacío si es Cq. 'clase' es opcional "
           "(solo para evaluar moléculas conocidas).")
plantilla = pd.DataFrame([
    {"delta_c": 18.0, "delta_h": 1.2, "mult": "CH3", "clase": "CH3"},
    {"delta_c": 58.0, "delta_h": 3.7, "mult": "CH2", "clase": "CH2-O"},
])
edited = st.data_editor(
    plantilla, num_rows="dynamic", use_container_width=True,
    column_config={
        "mult": st.column_config.SelectboxColumn(
            "mult", options=list(adapter.MULT_H.keys()), required=True),
        "clase": st.column_config.SelectboxColumn(
            "clase", options=[""] + list(class_names)),
    },
)

if st.button("Predecir", type="primary"):
    rows = edited.to_dict("records")
    peaks = []
    for r in rows:
        if r.get("delta_c") is None or (isinstance(r.get("delta_c"), float)
                                        and np.isnan(r["delta_c"])):
            continue
        dh = r.get("delta_h")
        dh = None if (dh is None or (isinstance(dh, float) and np.isnan(dh))) else float(dh)
        peaks.append({"delta_c": float(r["delta_c"]), "delta_h": dh,
                      "mult": r["mult"], "clase": r.get("clase") or None})

    fm = adapter.parse_formula(formula)
    inp = adapter.build_inputs(peaks, fm, norm)
    raw = predict_core.predict_raw(model, inp)
    total, ch2 = int(inp[4][0]), int(inp[4][1])
    cands = predict_core.candidatos(raw, fm, total, ch2, tau, k_max)

    st.subheader(f"Candidatos emitidos: K = {len(cands)}")
    df = pd.DataFrame({class_names[i]: [c[i] for c in cands]
                       for i in range(19)},
                      index=[f"cand {j}" + (" (ancla v2)" if j == 0 else "")
                             for j in range(len(cands))])
    st.dataframe(df.T, use_container_width=True)
    st.caption(f"crudo (redondeado): {list(np.round(raw, 2))}")

    have_clase = all(p["clase"] for p in peaks)
    if have_clase:
        yt = adapter.true_vector(peaks, class_names)
        cubierto = any(np.array_equal(yt, c) for c in cands)
        st.success("y_true CUBIERTO en K ✅") if cubierto else st.warning(
            "y_true NO cubierto en K ❌")
        diff = yt - cands[0]
        confus = [(class_names[i], int(diff[i])) for i in range(19) if diff[i] != 0]
        if confus:
            st.write("Diferencia ancla v2 vs verdadero (qué confunde):", confus)
    else:
        st.info("Sin columna 'clase' -> modo predicción real (sin evaluación).")
