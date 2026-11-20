import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.core.location import pooled_r
from app.ui import (COLORS, bias, check_yourself, explain, get_ds, loc_r, location_picker, predict_first, settings,
                    show, style, text)

ds = get_ds()
cfg = settings()
r_loc = loc_r()

st.title("Lab 1 · Meet the data")
st.caption("SEAS5 forecasts vs CHIRPS observations · Source: notebook §1")
st.markdown(text("lab1.intro", n_loc=f"{ds.n_locations:,}", n_years=ds.n_years))
predict_first("lab1.predict", r_raw=f"{pooled_r(ds):.2f}")

# --- 1A: scatter, raw vs anomalies ------------------------------------------------
st.markdown("#### ② Play: how closely does SEAS5 follow what fell?")
c1, c2 = st.columns([2, 3])
view = c1.radio("Show", ["Raw totals (mm)", "Anomalies: each location's own average removed"], key="lab1_view",
                help="An anomaly is how much wetter or drier a season was than that location's 25-year average.")
anom = view.startswith("Anomalies")
pts = np.random.default_rng(2026).choice(ds.n_locations, size=600, replace=False)   # the notebook's 600 points
X, Y = ds.X[:, pts], ds.Y[:, pts]
if anom:
    X, Y = X - ds.X[:, pts].mean(0), Y - ds.Y[:, pts].mean(0)
r_all = pooled_r(ds, anomalies=anom)

fig = go.Figure()
fig.add_trace(go.Scattergl(
    x=X.ravel(), y=Y.ravel(), mode="markers", name="One location in one season",
    marker=dict(size=4, color=COLORS["point"], opacity=0.25),
    hovertemplate="SEAS5 %{x:.0f} mm<br>CHIRPS %{y:.0f} mm<extra></extra>"))
lim = float(np.quantile(np.abs(np.r_[X.ravel(), Y.ravel()]), 0.995))
lo = -lim if anom else 0
fig.add_trace(go.Scatter(x=[lo, lim], y=[lo, lim], mode="lines", name="Perfect forecast (1:1)",
                         line=dict(color=COLORS["ink"], dash="dash", width=1.5), hoverinfo="skip"))
label = "anomaly (mm)" if anom else "OND total (mm)"
style(fig, f"SEAS5 {label}", f"CHIRPS {label}", height=470)
fig.update_xaxes(range=[lo, lim], constrain="domain")
fig.update_yaxes(range=[lo, lim], scaleanchor="x", scaleratio=1, constrain="domain")
c2.metric(f"Correlation r (all {ds.X.size:,} location-seasons)", f"{r_all:.2f}",
          delta=None if not anom else f"{r_all - pooled_r(ds):+.2f} vs raw totals", delta_color="off")
c2.caption("The plot shows the notebook's 600 random locations × 25 seasons; r uses every location.")
show(fig)

st.markdown("#### ③ Explain simply")
explain("lab1.scatter.explain", "lab1.scatter.maths", r_raw=f"{pooled_r(ds):.2f}", r_anom=f"{pooled_r(ds, True):.2f}")

# --- 1B: bias by amount -----------------------------------------------------------
st.divider()
st.markdown("#### Is the forecast's error the same everywhere?")
b = bias()
labels = [f"{lo_:.0f}–{hi_:.0f}" for lo_, hi_ in zip(b.edges[:-1], b.edges[1:])]
fig = go.Figure()
fig.add_trace(go.Bar(x=labels, y=b.q75 - b.q25, base=b.q25, name="Middle half of seasons (25th–75th percentile)",
                     marker=dict(color=COLORS["faint"], line=dict(width=0)), width=0.6,
                     hovertemplate="25th–75th percentile: %{base:.0f} to %{y:.0f} mm<extra></extra>"))
fig.add_trace(go.Scatter(x=labels, y=b.mean_bias, mode="lines+markers", name="Average bias (SEAS5 − CHIRPS)",
                         line=dict(color=COLORS["seas5"], width=2.5), marker=dict(size=9, line=dict(color="white", width=2)),
                         customdata=np.c_[b.mean_x, b.mean_y],
                         hovertemplate="Average bias %{y:+.1f} mm<br>SEAS5 avg %{customdata[0]:.0f} mm · "
                                       "CHIRPS avg %{customdata[1]:.0f} mm<extra></extra>"))
fig.add_hline(y=b.overall_mean, line=dict(color=COLORS["ink2"], dash="dot", width=1.5),
              annotation_text=f"One-number correction: {b.overall_mean:+.0f} mm", annotation_position="top left")
fig.add_hline(y=0, line=dict(color=COLORS["ink"], width=1))
style(fig, "SEAS5 forecast, grouped into tenths of all forecasts (mm)", "SEAS5 − CHIRPS (mm)", height=420)
show(fig)
explain("lab1.bias.explain", bias_lo=f"{b.mean_bias.min():+.0f}", bias_hi=f"{b.mean_bias.max():+.0f}",
        bias_all=f"{b.overall_mean:+.0f}")

# --- 1C: one location's story -----------------------------------------------------------
st.divider()
st.markdown("#### One location, 25 seasons")
c1, c2 = st.columns([1.3, 3])
loc, loc_label = location_picker(c1, "lab1")
x, y = ds.X[:, loc], ds.Y[:, loc]
r_here = r_loc[loc]
c1.metric("Year-to-year r here", "undefined" if np.isnan(r_here) else f"{r_here:.2f}")
c1.metric("Average CHIRPS", f"{y.mean():.0f} mm")
c1.metric("Average SEAS5", f"{x.mean():.0f} mm")

fig = go.Figure()
fig.add_vrect(x0=cfg.split_year + 0.5, x1=ds.years[-1] + 0.5, fillcolor=COLORS["idr"], opacity=0.06, line_width=0,
              annotation_text="Test years", annotation_position="top left")
fig.add_trace(go.Scatter(x=ds.years, y=y, mode="lines+markers", name="CHIRPS (what fell)",
                         line=dict(color=COLORS["chirps"], width=2), marker=dict(size=7),
                         hovertemplate="%{x}: CHIRPS %{y:.0f} mm<extra></extra>"))
fig.add_trace(go.Scatter(x=ds.years, y=x, mode="lines+markers", name="SEAS5 (forecast)",
                         line=dict(color=COLORS["seas5"], width=2.5), marker=dict(size=7, symbol="diamond"),
                         hovertemplate="%{x}: SEAS5 %{y:.0f} mm<extra></extra>"))
fig.add_hline(y=float(ds.wmo_mean[loc]), line=dict(color=COLORS["ink2"], dash="dot", width=1))
fig.add_annotation(x=ds.years[0] - 0.5, y=float(ds.wmo_mean[loc]), text=f"WMO reference mean {ds.wmo_mean[loc]:.0f} mm",
                   showarrow=False, xanchor="left", yanchor="bottom", font=dict(color=COLORS["ink2"], size=11),
                   bgcolor="rgba(252,252,251,0.85)")
for name, series, col in [("CHIRPS", y, COLORS["chirps"]), ("SEAS5", x, "#9a6a00")]:
    fig.add_annotation(x=ds.years[-1], y=series[-1], text=f"<b>{name}</b>", showarrow=False, xanchor="left",
                       xshift=8, font=dict(color=col))
style(fig, "Season (October–December)", "Rainfall total (mm)", height=400)
fig.update_xaxes(dtick=2, range=[ds.years[0] - 0.6, ds.years[-1] + 3.2])
fig.update_yaxes(rangemode="tozero")
c2.plotly_chart(fig, width="stretch", theme=None, config={"displaylogo": False})
explain("lab1.location.explain")

# --- 1D: distribution of per-location r ---------------------------------------------------
st.divider()
st.markdown("#### Every location's year-to-year match")
valid = r_loc[~np.isnan(r_loc)]
med = float(np.median(valid))
fig = go.Figure()
fig.add_trace(go.Histogram(x=valid, xbins=dict(start=-1, end=1, size=0.05), name="Locations",
                           marker=dict(color=COLORS["idr"], line=dict(color="#fcfcfb", width=1.5)),
                           hovertemplate="r %{x}<br>%{y} locations<extra></extra>"))
fig.add_vline(x=med, line=dict(color=COLORS["ink"], width=2),
              annotation_text=f"<b>Median r = {med:.2f}</b>", annotation_position="top right")
fig.add_vline(x=0, line=dict(color=COLORS["ink2"], dash="dot", width=1),
              annotation_text="No relationship", annotation_position="top left")
if not np.isnan(r_here):
    fig.add_vline(x=float(r_here), line=dict(color=COLORS["seas5"], width=2.5, dash="dash"),
                  annotation_text=f"This {loc_label}: {r_here:.2f}", annotation_position="top right",
                  annotation=dict(yshift=-22))
style(fig, "Correlation between SEAS5 and CHIRPS across the 25 seasons at one location", "Number of locations",
      height=380, showlegend=False)
counts_r, _ = np.histogram(valid, bins=np.arange(-1, 1.0001, 0.05))
fig.update_yaxes(range=[0, counts_r.max() * 1.3])
show(fig)
explain("lab1.r.explain", r_med=f"{med:.2f}", pct_neg=f"{(valid < 0).mean():.0%}",
        pct_weak=f"{(valid < 0.3).mean():.0%}", r_raw=f"{pooled_r(ds):.2f}")

check_yourself("lab1.check", bias_lo=f"{b.mean_bias.min():+.0f}", bias_hi=f"{b.mean_bias.max():+.0f}",
               bias_all=f"{b.overall_mean:+.0f}", r_med=f"{med:.2f}")
