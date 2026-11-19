"""ACMAD OND IDR learning app.  Run from the repository root:

    .venv/bin/streamlit run app/app.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="IDR learning lab", page_icon=":material/water_drop:", layout="wide")

from app.ui import apply_preset, sidebar_settings  # noqa: E402

nav = st.navigation({
    "": [st.Page("labs/home.py", title="Start here", icon=":material/home:", default=True)],
    "Learning path": [
        st.Page("labs/lab1_data.py", title="Lab 1 · Meet the data", icon=":material/scatter_plot:"),
        st.Page("labs/lab2_idr.py", title="Lab 2 · How IDR works", icon=":material/stairs:"),
        st.Page("labs/lab3_terciles.py", title="Lab 3 · From curve to terciles", icon=":material/stacked_bar_chart:"),
        st.Page("labs/lab4_scores.py", title="Lab 4 · Is it any good?", icon=":material/rule:"),
        st.Page("labs/lab5_traps.py", title="Lab 5 · Traps and illusions", icon=":material/warning:"),
    ],
    "Reference": [
        st.Page("labs/lab6_questions.py", title="Lab 6 · Question bank", icon=":material/quiz:"),
        st.Page("labs/glossary.py", title="Glossary", icon=":material/menu_book:"),
    ],
})
apply_preset()
sidebar_settings()
nav.run()
