"""Streamlit helpers shared by the labs: caching, settings, content, lab rhythm, chart styling."""
from __future__ import annotations

import re
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from app.core.data import load_data
from app.core.experiment import CATEGORIES, Settings, run_pooled, shuffle_experiment
from app.core.location import local_terciles, per_location_r, pooling_comparison
from app.core.traps import bias_by_amount, leakage_experiment, seed_sensitivity

CONTENT_FILE = Path(__file__).parent / "content" / "questions.md"

# Validated with the dataviz palette checker (light surface). Lighter colours fall
# below 3:1 contrast, so every use carries a direct text label.
COLORS = {
    "idr": "#2a78d6",        # IDR model (blue)
    "compare": "#e87ba4",    # comparison model, always dashed + labelled
    "BN": "#eb6834",         # dry pole
    "NN": "#a3a29c",         # neutral midpoint
    "AN": "#1baf7a",         # wet pole
    "point": "#8a8984",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "grid": "#e9e8e4",
    "seas5": "#eda100",      # SEAS5 forecast (yellow, always labelled directly)
    "chirps": "#0b0b0b",     # CHIRPS observations (ink)
    "faint": "#d4d3cd",
}
# Lab 5 tab labels, shared with the question bank so its links can open a specific tab.
TAB_A, TAB_B, TAB_C = "A · Pooling illusion", "B · Leakage", "C · Can calibration create skill?"
TAB_D, TAB_E = "D · Reliable ≠ skilful", "E · Not 1,500 independent cases"
LAB5_TABS = [TAB_A, TAB_B, TAB_C, TAB_D, TAB_E]
ORDINAL_BLUES = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordered SEAS5 values, light -> dark


# --- data & experiment (cached) ----------------------------------------------

@st.cache_resource(show_spinner="Loading data…")
def get_ds():
    return load_data()


@st.cache_resource(show_spinner="Fitting IDR…")
def _pooled(seed: int, n_train: int, n_test: int, split_year: int):
    return run_pooled(get_ds(), Settings(seed, n_train, n_test, split_year))


@st.cache_resource
def _local(split_year: int):
    ds = get_ds()
    return local_terciles(ds, ds.years <= split_year)


def _key(s: Settings) -> tuple:
    return (s.seed, s.n_train, s.n_test, s.split_year)


@st.cache_resource
def _per_loc_r():
    return per_location_r(get_ds())


@st.cache_resource
def _bias():
    return bias_by_amount(get_ds())


@st.cache_resource(show_spinner="Comparing the three pooling settings…")
def _pooling(key: tuple):
    return pooling_comparison(get_ds(), Settings(*key), common=True)


@st.cache_resource(show_spinner="Refitting IDR on 100 shuffled copies of SEAS5 (about 10 seconds, once)…")
def _shuffle(key: tuple):
    return shuffle_experiment(get_ds(), Settings(*key), n_shuffles=100)


@st.cache_resource(show_spinner="Redrawing the sample with 50 other random seeds…")
def _seeds(key: tuple):
    return seed_sensitivity(get_ds(), Settings(*key), n_seeds=50)


@st.cache_resource(show_spinner="Re-scoring…")
def _leak(key: tuple, local: bool, cut: bool, ref: bool, model: bool):
    return leakage_experiment(get_ds(), Settings(*key), local=local, leak_cutoffs=cut,
                              leak_reference=ref, leak_model=model)


def loc_r():
    return _per_loc_r()


def bias():
    return _bias()


def pooling():
    return _pooling(_key(settings()))


def shuffles():
    out = _shuffle(_key(settings()))
    st.session_state["_shuffle_done"] = _key(settings())
    return out


def shuffles_ready() -> bool:
    """True if this session already ran the shuffle experiment for the current settings."""
    return st.session_state.get("_shuffle_done") == _key(settings())


def seeds():
    return _seeds(_key(settings()))


def leak(local=False, cut=False, ref=False, model=False):
    return _leak(_key(settings()), bool(local), bool(cut), bool(ref), bool(model))


def settings() -> Settings:
    return st.session_state.get("settings", Settings())


def pooled():
    s = settings()
    return _pooled(s.seed, s.n_train, s.n_test, s.split_year)


def local():
    return _local(settings().split_year)


def sidebar_settings() -> None:
    d = Settings()
    with st.sidebar:
        st.markdown("### Experiment settings")
        if st.button("Reset to defaults", width="stretch"):
            for k in ("set_seed", "set_ntrain", "set_ntest", "set_split"):
                st.session_state.pop(k, None)
        seed = st.number_input("Random seed", 0, 10_000, d.seed, key="set_seed")
        n_train = st.slider("Training pairs", 200, 5000, d.n_train, step=100, key="set_ntrain")
        n_test = st.slider("Test pairs", 200, 5000, d.n_test, step=100, key="set_ntest")
        split = st.slider("Last training year", 2008, 2020, d.split_year, key="set_split")
        s = Settings(int(seed), int(n_train), int(n_test), int(split))
        st.session_state["settings"] = s
        if s.is_notebook_default:
            st.success("Default settings (same as the notebook)", icon=":material/check_circle:")
        else:
            st.warning("Custom settings: the numbers will differ from the defaults (and from the notebook).", icon=":material/tune:")
        st.caption(f"The model learns from {get_ds().years[0]}–{s.split_year} and is tested on "
                   f"{s.split_year + 1}–{get_ds().years[-1]}, using {s.n_train:,} random training pairs "
                   f"and {s.n_test:,} test pairs. The Home page explains the setup.")


def apply_preset() -> None:
    """Apply widget values queued by a "show me" button (Question bank), before any widget renders."""
    preset = st.session_state.pop("_preset", None)
    if preset:
        for k, v in preset.items():
            st.session_state[k] = v


def goto(label: str, page: str, key: str, **preset) -> None:
    """A button that opens another lab with some of its controls preset."""
    if st.button(label, key=key, icon=":material/arrow_forward:"):
        st.session_state["_preset"] = preset
        st.switch_page(page)


def category_picker(key: str, default: int = 2, label: str = "Category") -> int:
    names = {"BN": "BN · below normal", "NN": "NN · near normal", "AN": "AN · above normal"}
    c = st.radio(label, CATEGORIES, index=default, key=key, horizontal=True, format_func=names.get)
    return CATEGORIES.index(c)


WETNESS = {"Dry place (10th percentile)": 10, "Typical place (median)": 50, "Wet place (90th percentile)": 90}


def location_picker(col, prefix: str) -> tuple[int, str]:
    """Pick a location by how wet it is (mean OND rainfall percentile) or by column index."""
    from app.core.location import location_by_wetness
    ds = get_ds()
    pick = col.selectbox("Location", [*WETNESS, "Choose by column index"], index=1, key=f"{prefix}_loc",
                         help="Locations are ranked by their average October–December rainfall over 2001–2025.")
    if pick in WETNESS:
        return location_by_wetness(ds, WETNESS[pick]), pick.split(" (")[0].lower()
    loc = int(col.number_input("Column index", 0, ds.n_locations - 1, 0, key=f"{prefix}_idx"))
    return loc, f"column {loc}"


def sample_note(n_pairs: int, n_years: int, n_locs: int | None = None, extra: str = "") -> None:
    locs = f" · {n_locs:,} locations" if n_locs is not None else ""
    st.caption(f"Scored on **{n_pairs:,} test pairs** · **{n_years} seasons**{locs}. {extra}")


# --- content ------------------------------------------------------------------

@st.cache_data
def _content(mtime: float) -> dict[str, str]:
    text = re.sub(r"<!--.*?-->", "", CONTENT_FILE.read_text(encoding="utf-8"), flags=re.S)
    blocks: dict[str, str] = {}
    for part in re.split(r"^## ", text, flags=re.M)[1:]:
        key, _, body = part.partition("\n")
        blocks[key.strip()] = body.strip()
    return blocks


class _Keep(dict):
    def __missing__(self, k):
        return "{" + k + "}"


def text(key: str, **values) -> str:
    """Content block with {placeholders} filled from live values (unknown ones are left visible)."""
    raw = _content(CONTENT_FILE.stat().st_mtime).get(key, f"*[missing text: {key}]*")
    # Protect LaTeX braces: only replace {simple_names}.
    return re.sub(r"\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: str(_Keep(values)[m.group(1)]), raw)


# --- the lab rhythm -------------------------------------------------------------

def predict_first(key: str, **values) -> None:
    with st.container(border=True):
        st.markdown("**① Predict first**  \n" + text(f"{key}.q", **values))
        with st.expander("Reveal the answer"):
            st.markdown(text(f"{key}.a", **values))


def explain(key: str, maths_key: str | None = None, **values) -> None:
    st.markdown(text(key, **values))
    if maths_key:
        with st.expander("The maths"):
            st.markdown(text(maths_key, **values))


def _save_answer(key: str) -> None:
    st.session_state[f"_saved_{key}"] = st.session_state.get(key, "")


def answer_box(key: str, height: int = 90) -> str:
    """A text box whose content survives visiting other pages (Streamlit drops state of unrendered widgets)."""
    store = f"_saved_{key}"
    if key not in st.session_state and store in st.session_state:
        st.session_state[key] = st.session_state[store]
    value = st.text_area("Your answer (kept while this browser tab is open, across pages)", key=key,
                         height=height, on_change=_save_answer, args=(key,))
    st.session_state[store] = value
    return value


def versus_clim(v: float) -> str:
    """Plain-language comparison of a BSS with 0, generated from its sign."""
    if v > 0.005:
        return f"beats climatology (BSS {v:+.2f})"
    if v < -0.005:
        return f"is worse than climatology (BSS {v:+.2f})"
    return f"does no better than climatology (BSS {v:+.2f})"


def which_positive(vals) -> str:
    pos = [c for c, v in zip(CATEGORIES, vals) if v > 0]
    return {3: "all three terciles", 0: "none of the terciles"}.get(len(pos), " and ".join(pos) + " only")


def check_yourself(key: str, heading: bool = True, **values) -> None:
    if heading:
        st.divider()
        st.markdown("#### ④ Check yourself")
    with st.container(border=True):
        st.markdown(text(f"{key}.q", **values))
        answer_box(f"answer_{key}", height=90)
        with st.expander("Reveal an answer built from the numbers on your screen"):
            st.markdown(text(f"{key}.a", **values))


# --- charts ---------------------------------------------------------------------

def style(fig: go.Figure, x_title: str, y_title: str, height: int = 420, **layout) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=70, r=20, t=40, b=60),
        paper_bgcolor="#fcfcfb", plot_bgcolor="#fcfcfb",
        font=dict(color=COLORS["ink2"], size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="white", font_color=COLORS["ink"]),
        **layout,
    )
    axis = dict(showgrid=True, gridcolor=COLORS["grid"], zeroline=False, linecolor="#c3c2b7",
                ticks="outside", tickcolor="#c3c2b7", automargin=True)
    fig.update_xaxes(title=x_title, **axis)
    fig.update_yaxes(title=y_title, **axis)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", theme=None,
                    config={"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]})
