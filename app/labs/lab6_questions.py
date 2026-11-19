import numpy as np
import streamlit as st

from app.core.experiment import CATEGORIES
from app.core.location import pooled_r
from app.core.scores import bss, reliability, roc
from app.ui import (TAB_A, TAB_B, TAB_C, TAB_D, TAB_E, answer_box, bias, get_ds, goto, leak, loc_r, pooled, pooling, settings, shuffles, shuffles_ready,
                    text, versus_clim, which_positive)

ds, r = get_ds(), pooled()

st.title("Lab 6 · Question bank")
st.caption("Every discussion and take-home question from the practical, with the plot that answers it")
st.markdown(text("lab6.intro"))

L1, L2, L3 = "labs/lab1_data.py", "labs/lab2_idr.py", "labs/lab3_terciles.py"
L4, L5 = "labs/lab4_scores.py", "labs/lab5_traps.py"
A_OPTS = ["① Pooled model, pooled cutoffs (the notebook)", "② Same model, each location's own cutoffs",
          "③ Location-aware (anomaly) model, own cutoffs"]

SECTIONS = [
    ("Meet the data", "notebook §1", [
        ("bias_amount", [("Lab 1: bias by amount", L1, {})]),
        ("mean_bias", [("Lab 1: bias by amount", L1, {})]),
        ("why_distribution", [("Lab 2: the predictive CDF", L2, {"lab2_nb": False})]),
    ]),
    ("How IDR works", "notebook §2", [
        ("dry_threshold", [("Lab 2: four example forecasts", L2, {"lab2_nb": True})]),
        ("monotone", [("Lab 2: one threshold at a time", L2, {}), ("Lab 2: four example forecasts", L2, {"lab2_nb": True})]),
    ]),
    ("From curve to terciles", "notebook §3", [
        ("wmo_terciles", [("Lab 3: local cutoffs", L3, {"lab3_mode": "Local (this location)"})]),
    ]),
    ("Is the forecast any good?", "notebook §4", [
        ("auc_vs_bss", [("Lab 4: ROC", L4, {})]),
        ("auc_reliability", [("Lab 4: reliability", L4, {"lab4_cat": "AN"}), ("Lab 5: tab D", L5, {"lab5_tab": TAB_D})]),
        ("independent", [("Lab 4: stability", L4, {}), ("Lab 5: tab E", L5, {"lab5_tab": TAB_E})]),
        ("positive_bss", [("Lab 4: Brier score", L4, {})]),
        ("sparse_bins", [("Lab 4: reliability with 15 bins", L4, {"lab4_bins": 15})]),
        ("pooling_regional", [("Lab 5: tab A, local cutoffs", L5, {"lab5_a": A_OPTS[1], "lab5_tab": TAB_A}),
                              ("Lab 3: a dry place", L3, {"lab3_loc": "Dry place (10th percentile)",
                                                          "lab3_mode": "Pooled (one pair for all of Africa)"})]),
    ]),
    ("Take-home questions", "end of the notebook", [
        ("reliable_no_skill", [("Lab 5: tab D", L5, {"lab5_tab": TAB_D})]),
        ("clim_no_test", [("Lab 5: tab B, local, leaked reference", L5,
                           {"lab5_b_mode": "Local (anomaly model, own cutoffs)", "lab5_b_ref": True, "lab5_tab": TAB_B})]),
        ("create_predictability", [("Lab 5: tab C, run the shuffles", L5, {"lab5_c_run": True, "lab5_tab": TAB_C})]),
    ]),
]

# --- live numbers used in the answers -------------------------------------------------------
@st.cache_data(show_spinner="Collecting the numbers for the answers…")
def facts(cfg) -> dict:  # cfg is part of the cache key
    b, rl, pc = bias(), loc_r(), pooling()
    all_bss = bss(r.P, r.obs_cat, r.clim)
    aucs = [roc(r.P[:, j], r.obs_cat == j).auc for j in range(3)]
    sparse = []
    for j, c in enumerate(CATEGORIES):
        rel = reliability(r.P[:, j], (r.obs_cat == j).astype(float), 8)
        sparse.append(f"{c} {int((rel.count < 10).sum())} of {len(rel.count)}")
    honest, leaked = leak(True), leak(True, ref=True)
    model_leak = leak(True, model=True)
    valid = rl[~np.isnan(rl)]
    clim_test = [(r.obs_cat == j).mean() for j in range(3)]
    return dict(
        bias_lo=f"{b.mean_bias.min():+.0f}", bias_hi=f"{b.mean_bias.max():+.0f}", bias_all=f"{b.overall_mean:+.0f}",
        r_raw=f"{pooled_r(ds):.2f}", r_anom=f"{pooled_r(ds, True):.2f}", r_med=f"{np.median(valid):.2f}",
        pct_neg=f"{(valid < 0).mean():.0%}", q1=f"{r.q1:.0f}", q2=f"{r.q2:.0f}",
        bss_bn=f"{all_bss[0]:+.2f}", bss_nn=f"{all_bss[1]:+.2f}", bss_an=f"{all_bss[2]:+.2f}",
        auc_bn=f"{aucs[0]:.3f}", auc_nn=f"{aucs[1]:.3f}", auc_an=f"{aucs[2]:.3f}",
        sparse=" · ".join(sparse), n_test=f"{len(r.obs_cat):,}", n_years=r.sample.n_test_years,
        b1=f"{pc.pooled[0]:+.2f}", b2=f"{pc.pooled_local[0]:+.2f}", b3=f"{pc.anomaly_local[0]:+.2f}",
        b3_an=f"{pc.anomaly_local[2]:+.2f}",
        leak_honest=f"{honest.bss[0]:+.3f}", leak_ref=f"{leaked.bss[0]:+.3f}", leak_model=f"{model_leak.bss[0]:+.3f}",
        clim_bn=f"{r.clim[0]:.3f}", test_bn=f"{clim_test[0]:.3f}",
        pos_text=which_positive(all_bss), best_bss=CATEGORIES[int(np.argmax(all_bss))],
        best_auc=CATEGORIES[int(np.argmax(aucs))], worst_auc=CATEGORIES[int(np.argmin(aucs))],
        v2=versus_clim(pc.pooled_local[0]), v3=versus_clim(pc.anomaly_local[0]),
    )


def shuffle_facts() -> dict:
    sh = shuffles()
    return dict(shuf=f"{sh['shuffled'][:, 0].mean():+.3f}", wrong=f"{sh['wrong_direction'][:, 0].mean():+.3f}",
                const=f"{sh['constant_climatology'][0, 0]:+.3f}")


F = facts(settings())

n = 0
for title, source, items in SECTIONS:
    st.markdown(f"### {title}")
    st.caption(f"Source: {source}")
    for qid, links in items:
        n += 1
        with st.container(border=True):
            st.markdown(f"**Q{n}.** " + text(f"qb.{qid}.q", **F))
            cols = st.columns(len(links) + 2)
            for i, (label, page, preset) in enumerate(links):
                with cols[i]:
                    goto(label, page, key=f"go_{qid}_{i}", **preset)
            answer_box(f"qb_answer_{qid}", height=80)
            with st.expander("Reveal an answer built from the app's numbers"):
                if qid == "create_predictability" and not shuffles_ready():
                    st.markdown(text("qb.create_predictability.a_pending"))
                    if st.button("Run the shuffle experiment (refits IDR 200 times, about 5 seconds)",
                                 key="qb_run_shuffle"):
                        shuffles()
                        st.rerun()
                else:
                    st.markdown(text(f"qb.{qid}.a", **F, **(shuffle_facts() if qid == "create_predictability" else {})))
