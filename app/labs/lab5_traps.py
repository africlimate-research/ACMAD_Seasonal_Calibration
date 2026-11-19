import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.experiment import CATEGORIES
from app.core.scores import bss, murphy, reliability, year_bootstrap_bss
from app.core.traps import bss_by_year, pair_bootstrap_bss
from app.ui import (LAB5_TABS, COLORS, category_picker, check_yourself, explain, leak, pooled, pooling, predict_first,
                    sample_note, show, shuffles, style, text, versus_clim)

r = pooled()
s = r.sample
pc = pooling()
A_TEXT = dict(b1=f"{pc.pooled[0]:+.2f}", b2=f"{pc.pooled_local[0]:+.2f}", b3=f"{pc.anomaly_local[0]:+.2f}",
              v1=versus_clim(pc.pooled[0]), v2=versus_clim(pc.pooled_local[0]), v3=versus_clim(pc.anomaly_local[0]),
              ans3=("A little: consistent with the weak per-location correlations in Lab 1."
                    if pc.anomaly_local[0] > 0.005 else
                    "None that these settings can show: here even the location-aware model doesn't beat local "
                    "climatology. Small samples make this score very noisy (try the default settings)."))

st.title("Lab 5 · Traps and illusions")
st.caption("What the headline numbers do and don't mean · Source: notebook discussion and take-home questions")
st.markdown(text("lab5.intro"))
predict_first("lab5.predict", bss_bn=f"{bss(r.P, r.obs_cat, r.clim)[0]:+.2f}", **A_TEXT)

tabs = st.tabs(LAB5_TABS, key="lab5_tab", on_change="rerun")


def bss_bars(rows: dict[str, np.ndarray], highlight: str | None, height=380, y_title="Brier skill score"):
    """Grouped BSS bars: x = setting, one bar per category, highlighted setting at full strength."""
    fig = go.Figure()
    names = list(rows)
    for j, cat in enumerate(CATEGORIES):
        vals = [rows[n][j] for n in names]
        fig.add_trace(go.Bar(
            x=names, y=vals, name=cat,
            marker=dict(color=COLORS[cat], line=dict(color="#fcfcfb", width=2),
                        opacity=[1 if (highlight is None or n == highlight) else 0.22 for n in names]),
            text=[f"{cat} {v:+.2f}" for v in vals], textposition="outside", textfont=dict(color=COLORS["ink"], size=12),
            hovertemplate=f"{cat}: " + "%{y:+.3f}<extra>%{x}</extra>"))
    fig.add_hline(y=0, line=dict(color=COLORS["ink"], width=1.5),
                  annotation_text="0 = no better than climatology", annotation_position="bottom right")
    style(fig, "", y_title, height=height)
    allv = np.concatenate(list(rows.values()))
    fig.update_yaxes(range=[min(allv.min(), 0) - 0.12, max(allv.max(), 0) + 0.15])
    fig.update_layout(barmode="group", bargap=0.25, legend=dict(traceorder="normal"))
    return fig


# --- A. Pooling illusion -----------------------------------------------------------------
with tabs[0]:
    opts = {"① Pooled model, pooled cutoffs (the notebook)": pc.pooled,
            "② Same model, each location's own cutoffs": pc.pooled_local,
            "③ Location-aware (anomaly) model, own cutoffs": pc.anomaly_local}
    pick = st.radio("Setting", list(opts), key="lab5_a", horizontal=False)
    short = {k: k.split(" ", 1)[0] + " " + ["Pooled cutoffs", "Local cutoffs", "Anomaly model"][i]
             for i, k in enumerate(opts)}
    fig = bss_bars({short[k]: v for k, v in opts.items()}, short[pick])
    show(fig)
    sample_note(pc.n_test_pairs, s.n_test_years, None,
                f"All three settings score the same pairs: {pc.n_test_excluded} test pairs are left out because their "
                f"location's cutoffs are equal or its rainfall barely varies ({pc.n_locations_equal_cutoffs} locations "
                f"have equal cutoffs). The anomaly model is fitted on {pc.n_train_pairs_anomaly:,} training pairs.")
    st.info(f"Why is ① **{pc.pooled[0]:+.2f}** here and not the notebook's **{pc.pooled_all[0]:+.2f}**? The "
            f"{pc.n_test_excluded} left-out pairs are almost rainless places (median {pc.excluded_median_mm:.1f} mm "
            f"per season). With pooled cutoffs they are \"below normal\" nearly every year, and the pooled model "
            f"scores BSS **{pc.pooled_excluded[0]:+.2f}** for BN on them alone. Removing just these desert pairs "
            f"already shrinks the headline number.", icon=":material/info:")
    key = ["lab5.a.1", "lab5.a.2", "lab5.a.3"][list(opts).index(pick)]
    with st.container(border=True):
        st.markdown(text(key, **A_TEXT))
    explain("lab5.a.explain", "lab5.a.maths", **A_TEXT)

# --- B. Leakage --------------------------------------------------------------------------------
with tabs[1]:
    st.markdown(text("lab5.b.intro"))
    c1, c2 = st.columns([1.2, 2])
    mode = c1.radio("Model and cutoffs", ["Pooled (the notebook)", "Local (anomaly model, own cutoffs)"], key="lab5_b_mode")
    is_local = mode.startswith("Local")
    with c2:
        lc = st.toggle("Cutoffs use all 25 years (test years included)", key="lab5_b_cut")
        lr = st.toggle("Climatology reference uses all 25 years", key="lab5_b_ref")
        lm = st.toggle("IDR is also trained on the test pairs", key="lab5_b_model")
    honest, leaky = leak(is_local), leak(is_local, lc, lr, lm)
    any_leak = lc or lr or lm
    rows = {"Honest (training years only)": honest.bss}
    if any_leak:
        rows["With the leaks you switched on"] = leaky.bss
    show(bss_bars(rows, None if not any_leak else "With the leaks you switched on", height=340))
    sample_note(leaky.n_test, s.n_test_years)
    if any_leak:
        d = leaky.bss - honest.bss
        st.markdown(" · ".join(f"**{c}** {d[j]:+.3f}" for j, c in enumerate(CATEGORIES)) + " change in BSS. "
                    f"Brier score of the climatology reference: honest "
                    f"{', '.join(f'{v:.3f}' for v in honest.bs_ref)} → leaked {', '.join(f'{v:.3f}' for v in leaky.bs_ref)}.")
    explain("lab5.b.explain")

# --- C. Shuffles ---------------------------------------------------------------------------------
with tabs[2]:
    st.markdown(text("lab5.c.intro"))
    if st.toggle("Run the experiment (refits IDR 200 times; about 5 seconds the first time)", key="lab5_c_run"):
        j = category_picker("lab5_c_cat", default=0)
        cat = CATEGORIES[j]
        sh = shuffles()
        variants = [("real", "Real SEAS5"), ("shuffled", "Shuffled SEAS5<br>(right direction)"),
                    ("wrong_direction", "Shuffled SEAS5<br>(forced wrong direction)"),
                    ("shuffled_train_and_test", "Shuffled training<br>and test SEAS5"),
                    ("constant_climatology", "No SEAS5 at all<br>(constant climatology)")]
        fig = go.Figure()
        rng = np.random.default_rng(0)
        for i, (k, name) in enumerate(variants):
            v = sh[k][:, j]
            fig.add_trace(go.Scatter(x=i + rng.uniform(-0.18, 0.18, len(v)) * (len(v) > 1), y=v, mode="markers",
                                     marker=dict(size=7, color=COLORS["idr"] if k == "real" else COLORS["point"],
                                                 opacity=0.9 if len(v) == 1 else 0.45),
                                     showlegend=False, hovertemplate="BSS %{y:+.3f}<extra></extra>"))
            fig.add_annotation(x=i, y=v.mean(), text=f"<b>{v.mean():+.3f}</b>", showarrow=False, xanchor="left",
                               xshift=34, font=dict(color=COLORS["ink"]))
        fig.add_hline(y=0, line=dict(color=COLORS["ink"], width=1.5))
        style(fig, "", f"Brier skill score for {cat}", height=420, showlegend=False)
        fig.update_xaxes(tickvals=list(range(len(variants))), ticktext=[n for _, n in variants], showgrid=False)
        zoom = st.toggle("Zoom in on the shuffled results", value=True, key="lab5_c_zoom")
        if zoom:
            rest = np.concatenate([sh[k][:, j] for k, _ in variants[1:]])
            fig.update_yaxes(range=[rest.min() - 0.03, rest.max() + 0.03])
            fig.add_annotation(x=0, y=rest.max() + 0.02, text=f"Real: {sh['real'][0, j]:+.2f} (off the top) ↑",
                               showarrow=False, font=dict(color=COLORS["idr"]))
        show(fig)
        st.caption(f"Each grey dot is one of 100 shuffles; the number is the average. Test cases are fixed "
                   f"({len(s.y_test):,} pairs).")
        explain("lab5.c.explain", "lab5.c.maths",
                shuf=f"{sh['shuffled'][:, j].mean():+.3f}", wrong=f"{sh['wrong_direction'][:, j].mean():+.3f}",
                both=f"{sh['shuffled_train_and_test'][:, j].mean():+.3f}",
                const=f"{sh['constant_climatology'][0, j]:+.3f}", cat=cat)

# --- D. Reliable but not skilful ------------------------------------------------------------------
with tabs[3]:
    st.markdown(text("lab5.d.intro"))
    j = category_picker("lab5_d_cat", default=0)
    cat = CATEGORIES[j]
    o = (r.obs_cat == j).astype(float)
    fcs = {"IDR": r.P[:, j], "Always climatology": np.full(len(o), r.clim[j])}
    c1, c2 = st.columns([3, 2])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly reliable",
                             line=dict(color=COLORS["ink"], dash="dash", width=1.5), hoverinfo="skip"))
    rows = []
    for (name, f), col, sym in zip(fcs.items(), [COLORS["idr"], COLORS["compare"]], ["circle", "diamond"]):
        rel = reliability(f, o, 8)
        fig.add_trace(go.Scatter(x=rel.mean_p, y=rel.freq, mode="lines+markers", name=name,
                                 line=dict(color=col, width=2),
                                 marker=dict(size=13, symbol=[sym if c >= 10 else f"{sym}-open" for c in rel.count],
                                             line=dict(color=[("white" if c >= 10 else col) for c in rel.count], width=2)),
                                 customdata=rel.count,
                                 hovertemplate=f"{name}: " + "forecast %{x:.0%} → happened %{y:.0%} "
                                               "(%{customdata} forecasts)<extra></extra>"))
        m = murphy(f, o, 8)
        rows.append({"Forecast": name, "Reliability (lower = better)": m.reliability,
                     "Resolution (higher = better)": m.resolution, "Brier score": m.bs_raw,
                     "BSS": 1 - m.bs_raw / np.mean((r.clim[j] - o) ** 2)})
    style(fig, f"Forecast probability of {cat}", "How often it actually happened", height=430)
    fig.update_xaxes(range=[0, 1], tickformat=".0%", constrain="domain")
    fig.update_yaxes(range=[0, 1], tickformat=".0%", scaleanchor="x", scaleratio=1, constrain="domain")
    with c1:
        show(fig)
        st.caption("Hollow markers: bins with fewer than 10 forecasts, too few to judge.")
    with c2:
        st.dataframe(pd.DataFrame(rows).set_index("Forecast").T.style.format("{:.3f}"), width="stretch")
        st.markdown(f"Climatology always says **{r.clim[j]:.0%}**. In the test years {cat} happened "
                    f"**{o.mean():.1%}** of the time, so its single point sits "
                    f"{'just above' if o.mean() > r.clim[j] else 'just below'} the diagonal.")
    explain("lab5.d.explain", cat=cat, clim_p=f"{r.clim[j]:.1%}", obs_p=f"{o.mean():.1%}",
            gap=f"{abs(o.mean() - r.clim[j]) * 100:.1f} percentage points")

# --- E. Not 1,500 independent cases ---------------------------------------------------------------
with tabs[4]:
    st.markdown(text("lab5.e.intro", n_test=f"{len(s.y_test):,}"))
    c1, c2 = st.columns([1.2, 2])
    which = c1.radio("Model", ["Pooled (the notebook)", "Location-aware (anomaly model, own cutoffs)"], key="lab5_e_mode")
    with c2:
        j = category_picker("lab5_e_cat", default=0)
    cat = CATEGORIES[j]
    if which.startswith("Pooled"):
        P, cats, ref, yrs = r.P, r.obs_cat, np.broadcast_to(r.clim, r.P.shape), s.year_test
    else:
        L = leak(True)
        P, cats, ref, yrs = L.P, L.cat, L.ref, L.year
    now = bss(P, cats, ref)[j]
    pb = pair_bootstrap_bss(P, cats, ref)[:, j]
    yb = year_bootstrap_bss(P, cats, ref, yrs)[:, j]
    uy, per_year, n_py = bss_by_year(P, cats, ref, yrs)

    c1, c2 = st.columns(2)
    fig = go.Figure()
    for i, (name, vals) in enumerate([("Resample pairs<br>(treats every pair as independent)", pb),
                                      ("Resample whole seasons<br>(keeps each season's pairs together)", yb)]):
        lo, hi = np.percentile(vals, [5, 95])
        fig.add_trace(go.Scatter(x=[lo, hi], y=[i, i], mode="lines", line=dict(color=COLORS["idr"], width=10),
                                 showlegend=False, hovertemplate=f"90% range {lo:+.3f} to {hi:+.3f}<extra></extra>"))
        fig.add_annotation(x=hi, y=i, text=f"{lo:+.3f} to {hi:+.3f}", showarrow=False, xanchor="left", xshift=8,
                           font=dict(color=COLORS["ink"]))
    fig.add_vline(x=now, line=dict(color=COLORS["ink"], width=2), annotation_text=f"Score {now:+.3f}",
                  annotation_position="top")
    style(fig, f"BSS for {cat} (middle 90%)", "", height=260, showlegend=False)
    fig.update_yaxes(tickvals=[0, 1], ticktext=["Resample pairs", "Resample seasons"], range=[-0.6, 1.6], showgrid=False)
    span = max(pb.max(), yb.max()) - min(pb.min(), yb.min())
    fig.update_xaxes(range=[min(pb.min(), yb.min()) - 0.1 * span, max(pb.max(), yb.max()) + 0.45 * span])
    with c1:
        show(fig)
    fig = go.Figure(go.Bar(x=uy, y=per_year[:, j], marker=dict(color=COLORS[cat], line=dict(width=0)),
                           text=[f"{v:+.2f}" for v in per_year[:, j]], textposition="outside",
                           textfont=dict(color=COLORS["ink"]), customdata=n_py,
                           hovertemplate="%{x}: BSS %{y:+.3f} (%{customdata} pairs)<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=COLORS["ink"], width=1))
    style(fig, "Test season", f"BSS for {cat}, that season only", height=260, showlegend=False)
    pad = 0.25 * (per_year[:, j].max() - per_year[:, j].min() + 0.05)
    fig.update_yaxes(range=[min(0, per_year[:, j].min()) - pad, max(0, per_year[:, j].max()) + pad])
    fig.update_xaxes(dtick=1)
    with c2:
        show(fig)
    sample_note(len(cats), len(uy))
    explain("lab5.e.explain" if which.startswith("Pooled") else "lab5.e.explain_local", n_pairs=f"{len(cats):,}",
            n_years=len(uy),
            yr_lo=f"{per_year[:, j].min():+.2f}", yr_hi=f"{per_year[:, j].max():+.2f}",
            n_neg=int((per_year[:, j] < 0).sum()), now=f"{now:+.3f}", v_now=versus_clim(now))

# --- check yourself ---------------------------------------------------------------------------------
check_yourself("lab5.check", **A_TEXT)
