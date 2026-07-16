# coding: utf-8
"""
gui_inspector.py  -- corre en TU PC (no en el cluster)

Inspector visual de predicciones del modelo NMR.
Muestra: estructura 2D + vector real vs predicho (crudo y asistido),
carrusel por molecula, y filtros por tipo de error (incluye error direccional).

Requisitos (en tu PC):
    pip install streamlit pandas rdkit pyarrow numpy

Uso:
    streamlit run gui_inspector.py
Luego se abre en el navegador. Ajusta PRED_FILE si tu parquet tiene otro nombre.
"""
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

# --- CONFIG ---
PRED_FILE = "predictions_v10.parquet"   # el archivo que trajiste del cluster
CLASSES = ["CH3","CH2","CH","Cq","CH3-O","CH2-O","CH-O","Cq-O",
           "CH3-N","CH2-N","CH-N","Cq-N","=CH2","=CH/Ar","Cqsp2",
           "Aldeh","Imina","C-2X","C-3X"]
N = len(CLASSES)

st.set_page_config(page_title="NMR Inspector", layout="wide")


@st.cache_data
def load(path):
    try:
        df = pd.read_parquet(path)
    except Exception:
        df = pd.read_json(path.replace(".parquet", ".json"))
    # asegurar arrays numpy
    for col in ["y_true", "y_pred_crude", "y_pred_assisted"]:
        df[col] = df[col].apply(lambda v: np.array(v, dtype=int))
    return df


df = load(PRED_FILE)

st.title("NMR Inspector — real vs predicho")
st.caption(f"{len(df)} moleculas del set de validacion · archivo: {PRED_FILE}")

# ------------- SIDEBAR: modo y filtros -------------
st.sidebar.header("Modo de prediccion")
mode = st.sidebar.radio("¿Que prediccion mirar?",
                        ["y_pred_assisted", "y_pred_crude"],
                        format_func=lambda s: "Asistida (oraculo)" if "assist" in s else "Cruda (modelo solo)")

st.sidebar.header("Filtro por error")
filt = st.sidebar.selectbox(
    "Mostrar...",
    ["Todas",
     "Solo las que fallan (vector != real)",
     "Fallan en un grupo especifico",
     "Confusion direccional (X de mas, Y de menos)"])

# construir mascara segun filtro
pred_col = mode
diff = df[pred_col].apply(lambda p: p) - df["y_true"]  # no usado directo; calculamos por fila

def row_wrong(r):
    return not np.array_equal(r[pred_col], r["y_true"])

mask = pd.Series(True, index=df.index)

if filt == "Solo las que fallan (vector != real)":
    mask = df.apply(row_wrong, axis=1)

elif filt == "Fallan en un grupo especifico":
    g = st.sidebar.selectbox("Grupo", CLASSES)
    gi = CLASSES.index(g)
    mask = df.apply(lambda r: r[pred_col][gi] != r["y_true"][gi], axis=1)

elif filt == "Confusion direccional (X de mas, Y de menos)":
    col1, col2 = st.sidebar.columns(2)
    g_over  = col1.selectbox("Predice de MAS (X)", CLASSES, index=CLASSES.index("Cqsp2"))
    g_under = col2.selectbox("Predice de MENOS (Y)", CLASSES, index=CLASSES.index("=CH/Ar"))
    io, iu = CLASSES.index(g_over), CLASSES.index(g_under)
    mask = df.apply(
        lambda r: (r[pred_col][io] > r["y_true"][io]) and (r[pred_col][iu] < r["y_true"][iu]),
        axis=1)

sub = df[mask].reset_index(drop=True)
st.sidebar.metric("Moleculas que matchean el filtro", len(sub))

if len(sub) == 0:
    st.warning("Ninguna molecula matchea el filtro. Proba otro.")
    st.stop()

# ------------- CARRUSEL -------------
if "pos" not in st.session_state:
    st.session_state.pos = 0
st.session_state.pos = min(st.session_state.pos, len(sub) - 1)

c1, c2, c3, c4 = st.columns([1, 1, 3, 1])
if c1.button("◀ Anterior"):
    st.session_state.pos = (st.session_state.pos - 1) % len(sub)
if c2.button("Siguiente ▶"):
    st.session_state.pos = (st.session_state.pos + 1) % len(sub)
st.session_state.pos = c3.slider("Molecula", 0, len(sub) - 1, st.session_state.pos)
c4.write(f"{st.session_state.pos + 1} / {len(sub)}")

row = sub.iloc[st.session_state.pos]

# ------------- LAYOUT: estructura | vectores -------------
left, right = st.columns([1, 1.4])

with left:
    st.subheader("Estructura")
    st.code(row["smiles"], language=None)
    mol = Chem.MolFromSmiles(row["smiles"])
    if mol:
        img = Draw.MolToImage(mol, size=(420, 340))
        st.image(img)
    else:
        st.error("RDKit no pudo parsear este SMILES")
    st.caption(f"idx original en dataset: {row['idx']}")

with right:
    st.subheader(f"Vector real vs {'asistido' if 'assist' in mode else 'crudo'}")
    yt, yp = row["y_true"], row[pred_col]
    table = pd.DataFrame({
        "Grupo": CLASSES,
        "Real": yt,
        "Predicho": yp,
        "Δ": yp - yt,
    })
    # resaltar filas con error
    def hl(r):
        if r["Δ"] != 0:
            return ["background-color: #ffdddd"] * len(r)
        return [""] * len(r)
    st.dataframe(table.style.apply(hl, axis=1), height=560, use_container_width=True)

    perfect = np.array_equal(yt, yp)
    if perfect:
        st.success("✓ Prediccion PERFECTA (vector completo correcto)")
    else:
        errs = [(CLASSES[i], int(yp[i] - yt[i])) for i in range(N) if yp[i] != yt[i]]
        st.error("✗ Errores: " + ", ".join(f"{g} ({'+' if d>0 else ''}{d})" for g, d in errs))
