import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.experiment import CATEGORIES
from app.core.scores import brier, bss, confusion, murphy, reliability, roc, year_bootstrap_bss
from app.core.traps import bss_by_year
from app.ui import (COLORS, category_picker, check_yourself, explain, pooled, predict_first, sample_note, seeds, show,
                    style, text, which_positive)

r = pooled()
s = r.sample
n_years = s.n_test_years
n_locs = len(np.unique(s.loc_test))

st.title("Lab 4 · Is the forecast any good?")
st.caption("Brier score, reliability, ROC · Source: notebook §4")
st.markdown(text("lab4.intro"))
predict_first("lab4.predict")

st.markdown("#### ② Play")
c1, c2 = st.columns([2, 1])
with c1:
    j = category_picker("lab4_cat", default=2, label="Which event do you want to score?")
n_bins = c2.slider("Probability bins", 5, 15, 8, key="lab4_bins",
                   help="Forecasts are grouped into equal-width probability bins for the reliability diagram "
                        "and the decomposition. The notebook uses 8.")
cat = CATEGORIES[j]
p, o = r.P[:, j], (r.obs_cat == j).astype(float)
clim_p = np.full_like(p, r.clim[j])

# --- 4A Brier score and its three parts ------------------------------------------------
st.markdown("##### A · The Brier score: average squared miss")
bs_idr, bs_clim = brier(p, o), brier(clim_p, o)
m1, m2, m3 = st.columns(3)
m1.metric(f"Brier score, IDR ({cat})", f"{bs_idr:.3f}", help="Lower is better. 0 = perfect.")
m2.metric(f"Brier score, climatology ({cat})", f"{bs_clim:.3f}",
          help=f"Always forecasting the training-years frequency, {r.clim[j]:.3f}.")
m3.metric("Brier skill score (BSS)", f"{1 - bs_idr / bs_clim:+.3f}",
          help="1 − BS(IDR) / BS(climatology). Above 0 means better than climatology.")
sample_note(len(o), n_years, n_locs)

mi, mc = murphy(p, o, n_bins), murphy(clim_p, o, n_bins)
parts = ["Reliability<br>(penalty: probabilities<br>don't mean what they say)",
         "Resolution<br>(reward: forecasts change<br>when outcomes change)",
         "Uncertainty<br>(how hard the event is;<br>same for everyone)"]
fig = go.Figure()
for name, m, col in [("IDR", mi, COLORS["idr"]), ("Climatology", mc, COLORS["compare"])]:
    vals = [m.reliability, m.resolution, m.uncertainty]
    fig.add_trace(go.Bar(x=parts, y=vals, name=name, marker=dict(color=col, line=dict(color="#fcfcfb", width=2)),
                         text=[f"{name} {v:.4f}" for v in vals], textposition="outside",
                         textfont=dict(color=COLORS["ink"], size=12),
                         hovertemplate=f"{name}: " + "%{y:.4f}<extra></extra>"))
style(fig, "", "Contribution to the Brier score", height=380)
fig.update_layout(barmode="group", bargap=0.3)
fig.update_yaxes(range=[0, max(mi.uncertainty, mc.uncertainty) * 1.25])
show(fig)
st.markdown(
    f"**IDR:** binned BS = REL − RES + UNC = {mi.reliability:.4f} − {mi.resolution:.4f} + {mi.uncertainty:.4f} "
    f"= **{mi.bs_binned:.4f}**. The raw BS is {mi.bs_raw:.4f}; the difference ({mi.within_bin:+.4f}) comes from "
    f"rounding each forecast to its bin's average.  \n"
    f"**Climatology:** REL {mc.reliability:.4f}, RES {mc.resolution:.4f}, UNC {mc.uncertainty:.4f}.")
explain("lab4.brier.explain", "lab4.brier.maths")

# --- 4B Reliability diagram -----------------------------------------------------------------
st.divider()
st.markdown("##### B · Reliability: when IDR says 70%, does it happen 70% of the time?")
rel = reliability(p, o, n_bins)
sparse = rel.count < 10
c1, c2 = st.columns([3, 2])
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly reliable",
                         line=dict(color=COLORS["ink"], dash="dash", width=1.5), hoverinfo="skip"))
fig.add_hline(y=o.mean(), line=dict(color=COLORS["ink2"], dash="dot", width=1),
              annotation_text=f"How often {cat} happened overall ({o.mean():.0%})", annotation_position="bottom right")
fig.add_trace(go.Scatter(x=rel.mean_p, y=rel.freq, mode="lines", line=dict(color=COLORS["idr"], width=2),
                         showlegend=False, hoverinfo="skip"))
for flag, name, sym in [(False, "IDR bin (10+ forecasts)", "circle"), (True, "Sparse bin (< 10 forecasts)", "circle-open")]:
    m = sparse == flag
    fig.add_trace(go.Scatter(
        x=rel.mean_p[m], y=rel.freq[m], mode="markers", name=name,
        marker=dict(size=12, symbol=sym, color=COLORS["point"] if flag else COLORS["idr"],
                    line=dict(color=COLORS["point"] if flag else "white", width=2)),
        customdata=rel.count[m],
        hovertemplate="Forecast %{x:.0%} → happened %{y:.0%}<br>%{customdata} forecasts<extra></extra>"))
style(fig, f"Forecast probability of {cat}", "How often it actually happened", height=440)
fig.update_xaxes(range=[0, 1], tickformat=".0%", constrain="domain")
fig.update_yaxes(range=[0, 1], tickformat=".0%", scaleanchor="x", scaleratio=1, constrain="domain")
with c1:
    show(fig)

edges = rel.edges
k = np.clip(np.floor(p * n_bins).astype(int), 0, n_bins - 1)
counts = np.bincount(k, minlength=n_bins)
fig = go.Figure(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=0.9 / n_bins,
                       marker=dict(color=[COLORS["point"] if 0 < c < 10 else COLORS["idr"] for c in counts],
                                   line=dict(width=0)),
                       hovertemplate="%{y} forecasts<extra>%{x:.0%}</extra>"))
style(fig, f"Forecast probability of {cat}", "Number of forecasts", height=440, showlegend=False)
fig.update_xaxes(range=[0, 1], tickformat=".0%")
with c2:
    st.markdown("**Sharpness: how often each probability was issued**")
    show(fig)
explain("lab4.reliability.explain", n_sparse=int(sparse.sum()), n_bins_used=len(rel.count))

# --- 4C ROC ---------------------------------------------------------------------------------
st.divider()
st.markdown("##### C · Discrimination (ROC): can IDR tell event seasons from the rest?")
cut = st.slider(f"Issue a {cat} warning when P({cat}) is at least", 0.0, 1.0, 0.5, 0.01, key="lab4_cut")
c1, c2 = st.columns([3, 2])
fig = go.Figure()
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance (AUC 0.5)",
                         line=dict(color=COLORS["ink2"], dash="dot", width=1), hoverinfo="skip"))
aucs = {}
for jj, cc in enumerate(CATEGORIES):
    rc = roc(r.P[:, jj], r.obs_cat == jj)
    if rc is None:
        continue
    aucs[cc] = rc.auc
    fig.add_trace(go.Scatter(x=rc.fpr, y=rc.tpr, mode="lines", name=f"{cc} (AUC {rc.auc:.3f})",
                             line=dict(color=COLORS[cc], width=3.5 if jj == j else 1.5),
                             opacity=1 if jj == j else 0.55,
                             hovertemplate=f"{cc}: " + "false alarm rate %{x:.2f}, hit rate %{y:.2f}<extra></extra>"))
cm = confusion(p, o, cut)
hr = cm["hits"] / max(cm["hits"] + cm["misses"], 1)
far = cm["false_alarms"] / max(cm["false_alarms"] + cm["correct_negatives"], 1)
fig.add_trace(go.Scatter(x=[far], y=[hr], mode="markers", name=f"Your cutoff ({cut:.0%})",
                         marker=dict(size=14, color=COLORS[cat], line=dict(color=COLORS["ink"], width=2)),
                         hovertemplate="Hit rate %{y:.2f}<br>False alarm rate %{x:.2f}<extra></extra>"))
style(fig, "False alarm rate (share of non-events warned)", "Hit rate (share of events warned)", height=440)
fig.update_xaxes(range=[0, 1], constrain="domain")
fig.update_yaxes(range=[0, 1], scaleanchor="x", scaleratio=1, constrain="domain")
with c1:
    show(fig)
with c2:
    st.markdown(f"**At a {cut:.0%} cutoff, for {cat}:**")
    st.dataframe(pd.DataFrame({f"{cat} happened": [cm["hits"], cm["misses"]],
                               f"{cat} did not happen": [cm["false_alarms"], cm["correct_negatives"]]},
                              index=["Warned", "Not warned"]), width="stretch")
    st.markdown(f"Hits **{cm['hits']}**, misses **{cm['misses']}**, false alarms **{cm['false_alarms']}**, "
                f"correct quiet **{cm['correct_negatives']}**.  \nHit rate **{hr:.2f}** · false alarm rate **{far:.2f}**.")
explain("lab4.roc.explain", "lab4.roc.maths")

# --- 4D Stability -----------------------------------------------------------------------------
st.divider()
st.markdown("##### D · How much would these numbers move with different luck?")
st.markdown(text("lab4.stability.intro"))
bss_now = bss(r.P, r.obs_cat, r.clim)[j]
sd = seeds()[:, j]
yb = year_bootstrap_bss(r.P, r.obs_cat, r.clim, s.year_test)[:, j]
uy, per_year, n_per_year = bss_by_year(r.P, r.obs_cat, r.clim, s.year_test)

c1, c2 = st.columns(2)
fig = go.Figure()
for i, (name, vals, help_) in enumerate([
        ("Different random pairs<br>(50 other seeds)", sd, "pair-sampling"),
        ("Different mix of seasons<br>(500 resamples of the 10 test years)", yb, "held-out-season")]):
    lo_, hi_ = np.percentile(vals, [5, 95])
    fig.add_trace(go.Scatter(x=[lo_, hi_], y=[i, i], mode="lines", line=dict(color=COLORS["idr"], width=10),
                             showlegend=False, hovertemplate=f"90% range {lo_:.3f} to {hi_:.3f}<extra></extra>"))
    fig.add_annotation(x=hi_, y=i, text=f"{lo_:.2f} to {hi_:.2f}", showarrow=False, xanchor="left", xshift=8,
                       font=dict(color=COLORS["ink"]))
fig.add_vline(x=bss_now, line=dict(color=COLORS["ink"], width=2), annotation_text=f"This run {bss_now:+.3f}",
              annotation_position="top")
style(fig, f"BSS for {cat} (middle 90% of reruns)", "", height=260, showlegend=False)
fig.update_yaxes(tickvals=[0, 1], ticktext=["Different random pairs<br>(50 other seeds)",
                                            "Different mix of seasons<br>(resampled test years)"],
                 range=[-0.6, 1.6], showgrid=False)
lo_v, hi_v = min(np.percentile(sd, 5), np.percentile(yb, 5), bss_now), max(np.percentile(sd, 95), np.percentile(yb, 95), bss_now)
span = hi_v - lo_v
fig.update_xaxes(range=[lo_v - 0.15 * span, hi_v + 0.9 * span])
fig.update_layout(xaxis_title=f"BSS for {cat}: middle 90% of reruns")
with c1:
    show(fig)

fig = go.Figure(go.Bar(x=uy, y=per_year[:, j], marker=dict(color=COLORS[cat], line=dict(width=0)), name="One season",
                       customdata=n_per_year, text=[f"{v:+.2f}" for v in per_year[:, j]], textposition="inside",
                       insidetextanchor="start", textangle=0, textfont=dict(color=COLORS["ink"], size=10),
                       hovertemplate="%{x}: BSS %{y:+.3f} (%{customdata} pairs)<extra></extra>"))
fig.add_trace(go.Scatter(x=[uy[0] - 0.5, uy[-1] + 0.5], y=[bss_now, bss_now], mode="lines", name="All seasons together",
                         line=dict(color=COLORS["ink"], dash="dash", width=1.5), hoverinfo="skip"))
fig.add_hline(y=0, line=dict(color=COLORS["ink2"], width=1))
style(fig, "Test season", f"BSS for {cat}, that season only", height=260)
fig.update_xaxes(dtick=1)
fig.update_yaxes(range=[min(0, per_year[:, j].min()) - 0.1, max(per_year[:, j].max(), bss_now) + 0.15])
with c2:
    show(fig)
worst = uy[np.argmin(per_year[:, j])]
explain("lab4.stability.explain", cat=cat, worst=worst, worst_bss=f"{per_year[:, j].min():+.2f}",
        best_bss=f"{per_year[:, j].max():+.2f}", n_years=n_years)

# --- check yourself ---------------------------------------------------------------------------
all_bss = bss(r.P, r.obs_cat, r.clim)
vals = dict(
    bss_bn=f"{all_bss[0]:+.2f}", bss_nn=f"{all_bss[1]:+.2f}", bss_an=f"{all_bss[2]:+.2f}",
    auc_bn=f"{aucs.get('BN', np.nan):.3f}", auc_nn=f"{aucs.get('NN', np.nan):.3f}", auc_an=f"{aucs.get('AN', np.nan):.3f}",
    best_auc=max(aucs, key=aucs.get), worst_auc=min(aucs, key=aucs.get), best_bss=CATEGORIES[int(np.argmax(all_bss))],
    pos_text=which_positive(all_bss), n_test=f"{len(o):,}",
    n_years=n_years,
    sparse=" · ".join(f"{c} {int((reliability(r.P[:, jj], (r.obs_cat == jj).astype(float), 8).count < 10).sum())}"
                      for jj, c in enumerate(CATEGORIES)),
)
check_yourself("lab4.check1", **vals)
check_yourself("lab4.check2", heading=False, **vals)
