import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.core.experiment import CATEGORIES, categories, cdf_at_rows, predictive_cdf, predictive_quantiles, tercile_probs
from app.ui import COLORS, check_yourself, location_picker, explain, get_ds, local, pooled, predict_first, show, style, text

ds, r, lt = get_ds(), pooled(), local()
s, model = r.sample, r.model
train_rows = s.train_rows

st.title("Lab 3 · From curve to terciles")
st.caption("P(BN), P(NN), P(AN) · Source: notebook §3")
st.markdown(text("lab3.intro"))
predict_first("lab3.predict")

# --- controls -----------------------------------------------------------------
st.markdown("#### ② Play")
c1, c2, c3 = st.columns([1.3, 1.3, 2])
mode = c1.radio("Tercile cutoffs", ["Pooled (one pair for all of Africa)", "Local (this location)"], key="lab3_mode",
                help="Pooled: one pair of cutoffs for all of Africa. Local: this location's own training-year terciles.")
loc, loc_label = location_picker(c2, "lab3")
x_loc_tr, y_loc_tr = ds.X[train_rows, loc], ds.Y[train_rows, loc]
x_max = int(max(np.quantile(s.x_train, 0.99), x_loc_tr.max() * 1.2, 50))
x_val = c3.slider("SEAS5 forecast for the season (mm)", 0, x_max, int(round(x_loc_tr.mean())), step=1,
                  key=f"lab3_x_{loc}",
                  help="Starts at this location's average SEAS5 forecast in the training years.")
c3.caption(f"Location column **{loc}**: training-year CHIRPS mean **{y_loc_tr.mean():.0f} mm**, "
           f"SEAS5 mean **{x_loc_tr.mean():.0f} mm**.")

local_mode = mode.startswith("Local")
if local_mode and not lt.distinct[loc]:
    st.warning("This location's lower and upper cutoffs are equal (too many identical seasons, e.g. all 0 mm), "
               "so local terciles are undefined. Showing pooled cutoffs instead.", icon=":material/warning:")
    local_mode = False
q1, q2 = (float(lt.q1[loc]), float(lt.q2[loc])) if local_mode else (r.q1, r.q2)
clim = lt.reference[loc] if local_mode else r.clim

thr, cdf = predictive_cdf(model, x_val)
F1, F2 = (float(v) for v in cdf_at_rows(model.predict(np.array([x_val, x_val])), np.array([q1, q2])))
P = np.array([F1, F2 - F1, 1 - F2])

# --- 3A: curve with cutoffs ----------------------------------------------------
upper = float(max(predictive_quantiles(model, x_val, (0.97,))[0], q2 * 1.4, y_loc_tr.max() * 1.1, 30))
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.r_[0, thr], y=np.r_[0, cdf], mode="lines", line_shape="hv", name="IDR predictive CDF",
                         line=dict(color=COLORS["idr"], width=2.5),
                         hovertemplate="P(CHIRPS ≤ %{x:.0f} mm) = %{y:.2f}<extra></extra>"))
fig.add_trace(go.Scatter(x=y_loc_tr, y=np.full(len(y_loc_tr), 0.015), mode="markers",
                         name="This location's 15 training seasons (CHIRPS)",
                         marker=dict(symbol="line-ns", size=14, line=dict(width=2, color=COLORS["ink2"])),
                         hovertemplate="%{x:.0f} mm<extra></extra>"))
for q, F, lab in [(q1, F1, "q⅓"), (q2, F2, "q⅔")]:
    fig.add_vline(x=q, line=dict(color=COLORS["ink"], dash="dash", width=1.5),
                  annotation_text=f"{lab} = {q:.0f} mm", annotation_position="top right")
    fig.add_shape(type="line", x0=0, x1=q, y0=F, y1=F, line=dict(color=COLORS["ink2"], dash="dot", width=1))
    fig.add_trace(go.Scatter(x=[q], y=[F], mode="markers", showlegend=False,
                             marker=dict(size=10, color=COLORS["idr"], line=dict(color="white", width=2)),
                             hovertemplate=f"F({lab}) = {F:.2f}<extra></extra>"))
for (lo, hi), cat, p in zip([(0, F1), (F1, F2), (F2, 1)], CATEGORIES, P):
    fig.add_shape(type="rect", xref="x domain", x0=0, x1=0.022, y0=lo, y1=hi,
                  fillcolor=COLORS[cat], line=dict(color="#fcfcfb", width=2))
    if hi - lo > 0.04:
        fig.add_annotation(xref="x domain", x=0.03, y=(lo + hi) / 2, text=f"<b>P({cat})</b> {p:.0%}",
                           showarrow=False, xanchor="left", font=dict(color=COLORS["ink"]))
style(fig, "Rainfall (mm)", "P(CHIRPS ≤ rainfall)", height=440)
fig.update_xaxes(range=[0, upper])
fig.update_yaxes(range=[0, 1.02])
show(fig)

# --- 3B: stacked bars ------------------------------------------------------------
def stacked(rows: dict[str, np.ndarray]) -> go.Figure:
    f = go.Figure()
    for j, cat in enumerate(CATEGORIES):
        vals = [v[j] for v in rows.values()]
        f.add_trace(go.Bar(y=list(rows), x=vals, orientation="h", name=cat,
                           marker=dict(color=COLORS[cat], line=dict(color="#fcfcfb", width=2)),
                           text=[f"{cat} {v:.0%}" if v >= 0.06 else "" for v in vals], textposition="inside",
                           insidetextanchor="middle", textfont=dict(color=COLORS["ink"]),
                           hovertemplate=f"{cat}: " + "%{x:.1%}<extra>%{y}</extra>"))
    f.update_layout(barmode="stack", bargap=0.35)
    return f

fig = stacked({"Climatology (training years)": clim, "This forecast": P})
style(fig, "Probability", "", height=200)
fig.update_xaxes(range=[0, 1], tickformat=".0%")
fig.update_layout(legend=dict(traceorder="normal"))
show(fig)

st.markdown("#### ③ Explain simply")
explain("lab3.cdf.explain", "lab3.cdf.maths", q1_mm=f"{q1:.0f}", q2_mm=f"{q2:.0f}")

# --- the twist: pooled vs local at one location --------------------------------
st.divider()
st.markdown("#### Why *which* cutoffs matters: every held-out season at this location")
with st.container(border=True):
    st.markdown(text("lab3.local.explain"))

years_te = ds.years[s.test_rows]
x_te, y_te = ds.X[s.test_rows, loc], ds.Y[s.test_rows, loc]
P_te = tercile_probs(model, x_te, q1, q2)
obs = categories(y_te, q1, q2)
hit = obs == P_te.argmax(1)
fig = go.Figure()
for j, cat in enumerate(CATEGORIES):
    fig.add_trace(go.Bar(x=years_te, y=P_te[:, j], name=f"P({cat})",
                         marker=dict(color=COLORS[cat], line=dict(color="#fcfcfb", width=2)),
                         customdata=np.c_[x_te, y_te],
                         hovertemplate=f"P({cat}) " + "%{y:.0%}<br>SEAS5 %{customdata[0]:.0f} mm · "
                                       "CHIRPS %{customdata[1]:.0f} mm<extra>%{x}</extra>"))
fig.add_trace(go.Scatter(x=years_te, y=np.full(len(years_te), 1.07), mode="text", showlegend=False,
                         text=[f"<b>{CATEGORIES[o]}</b>{' ✓' if h else ''}" for o, h in zip(obs, hit)],
                         textfont=dict(color=COLORS["ink"]), hoverinfo="skip"))
style(fig, "Held-out season", "Forecast probability", height=330)
fig.update_layout(barmode="stack", bargap=0.25, legend=dict(traceorder="normal"))
fig.update_yaxes(range=[0, 1.14], tickformat=".0%")
fig.update_xaxes(dtick=1)
show(fig)
st.caption(f"Text above each bar: the category that actually happened (✓ = the most likely category was right). "
           f"Right in **{hit.sum()} of {len(hit)}** seasons with **{'local' if local_mode else 'pooled'}** cutoffs. "
           "Switch the cutoffs above and compare.")

# --- 3C: notebook table ------------------------------------------------------------
st.divider()
st.markdown("#### Across all held-out forecasts (pooled cutoffs)")
st.markdown(text("lab3.table.explain", n_test=f"{len(s.y_test):,}"))
tbl = pd.DataFrame({
    "Category": CATEGORIES,
    "Average forecast probability": r.P.mean(0),
    "Observed frequency": [(r.obs_cat == j).mean() for j in range(3)],
})
st.dataframe(tbl.style.format({"Average forecast probability": "{:.3f}", "Observed frequency": "{:.3f}"}),
             hide_index=True, width="content")
st.caption(f"Pooled cutoffs: BN ≤ {r.q1:.1f} mm; AN > {r.q2:.1f} mm. Source: notebook §3, cell 13.")

# --- check yourself ------------------------------------------------------------------
def counts(c):
    return " · ".join(f"{cat} {int((c == j).sum())}" for j, cat in enumerate(CATEGORIES))

pooled_c = categories(y_loc_tr, r.q1, r.q2)
local_c = categories(y_loc_tr, lt.q1[loc], lt.q2[loc]) if lt.distinct[loc] else None
check_yourself("lab3.check", loc_label=loc_label, pooled_counts=counts(pooled_c),
               local_counts=counts(local_c) if local_c is not None else "undefined (equal cutoffs)")
