import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.core.experiment import (LinearGaussian, cdf_at_rows, fit_idr, predictive_cdf, predictive_quantiles,
                                 threshold_curve)
from app.core.pava import expand, group_by_x, pava_steps_antitonic
from app.ui import COLORS, ORDINAL_BLUES, check_yourself, explain, predict_first, pooled, show, style, text

r = pooled()
s, model = r.sample, r.model
x_hi = float(np.quantile(s.x_train, 0.99))
y_hi = float(np.quantile(s.y_train, 0.98))

st.title("Lab 2 · How IDR works")
st.caption("The core idea of the practical · Source: notebook §2")
st.markdown(text("lab2.intro"))
predict_first("lab2.predict", t_mm=f"{r.q1:.0f}")

# ---------------------------------------------------------------------------
st.markdown("#### ② Play: one threshold at a time")
c1, c2 = st.columns([3, 1])
t = c1.slider("Rainfall threshold *t* (mm)", 0, int(y_hi), int(round(r.q1)), step=5, key="lab2_t",
              help=f"Starts at the pooled lower tercile q⅓ = {r.q1:.0f} mm, which Lab 3 uses.")
log_x = c2.toggle("Log SEAS5 axis", value=True, key="lab2_log",
                  help="Half the locations get under 50 mm, so a log axis spreads out the dry end.")

below = s.y_train <= t
jitter = np.random.default_rng(0).uniform(-0.05, 0.05, len(below))
xs, p_fit = threshold_curve(model, s.x_train, t)
grid = np.geomspace(max(xs.min(), 1.0), xs.max(), 400) if log_x else np.linspace(xs.min(), xs.max(), 400)
p_grid = cdf_at_rows(model.predict(grid), t)
xplot = np.maximum(s.x_train, 1.0) if log_x else s.x_train

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=xplot, y=below + jitter, mode="markers", name="Past seasons (1 = stayed ≤ t, 0 = wetter)",
    marker=dict(size=6, color=COLORS["point"], opacity=0.35, line=dict(width=0)),
    customdata=s.y_train, hovertemplate="SEAS5 %{x:.0f} mm<br>CHIRPS %{customdata:.0f} mm<extra></extra>"))
fig.add_trace(go.Scatter(
    x=grid, y=p_grid, mode="lines", name="IDR: P(CHIRPS ≤ t | SEAS5)",
    line=dict(color=COLORS["idr"], width=2.5),
    hovertemplate="SEAS5 %{x:.0f} mm<br>P(≤ t) = %{y:.2f}<extra>IDR</extra>"))
fig.add_annotation(x=np.log10(grid[-1]) if log_x else grid[-1], y=p_grid[-1], text="IDR fit",
                   showarrow=False, xanchor="right", yanchor="bottom", font=dict(color=COLORS["ink2"]))
style(fig, "SEAS5 OND forecast (mm)", f"Stayed at or below {t} mm?", height=430)
fig.update_yaxes(range=[-0.1, 1.1], tickvals=[0, 0.25, 0.5, 0.75, 1])
if log_x:
    fig.update_xaxes(type="log")
show(fig)

dry_x, wet_x = np.quantile(s.x_train, [0.1, 0.9])
p_dry, p_wet = cdf_at_rows(model.predict(np.array([dry_x, wet_x])), t)
m1, m2, m3 = st.columns(3)
m1.metric("All training seasons ≤ t", f"{below.mean():.0%}", help="Climatology: ignores SEAS5 completely.")
m2.metric(f"IDR, dry forecast ({dry_x:.0f} mm)", f"{p_dry:.0%}")
m3.metric(f"IDR, wet forecast ({wet_x:.0f} mm)", f"{p_wet:.0%}")

st.markdown("#### ③ Explain simply")
explain("lab2.threshold.explain", "lab2.threshold.maths")

# ---------------------------------------------------------------------------
st.divider()
st.markdown("#### Step inside: how the curve is forced downward (PAVA)")
st.markdown(text("lab2.pava.explain"))

order = np.argsort(s.x_train)
pick = order[np.linspace(0, len(order) - 1, 30).astype(int)]
px, py = s.x_train[pick], s.y_train[pick]
ux, raw, w = group_by_x(px, py <= t)
steps = pava_steps_antitonic(raw, w)
check = threshold_curve(fit_idr(px, py), px, t)[1]   # IDRModel on the same 30 points
assert np.allclose(expand(steps[-1], len(ux)), check, atol=1e-3)

if len(steps) <= 2:
    st.info(f"At t = {t} mm these 30 seasons are already in order, so nothing needs pooling. "
            "Move the threshold to see PAVA at work.", icon=":material/info:")
k = st.slider("PAVA step", 0, len(steps) - 1, 0, key=f"lab2_pava_{t}") if len(steps) > 1 else 0
step = steps[k]
fitted = expand(step, len(ux))
pos = np.arange(len(ux)) + 1

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=pos, y=raw, mode="markers", name="Raw share ≤ t at this SEAS5 value",
    marker=dict(size=10, color=COLORS["point"], symbol="circle-open", line=dict(width=2)),
    customdata=ux, hovertemplate="SEAS5 %{customdata:.0f} mm<br>raw share %{y:.2f}<extra></extra>"))
for b in step.blocks:
    hl = step.merged is not None and (b.start, b.stop) == step.merged
    fig.add_trace(go.Scatter(
        x=[b.start + 0.6, b.stop + 0.4], y=[b.value, b.value], mode="lines", showlegend=False,
        line=dict(color=COLORS["ink"] if hl else COLORS["idr"], width=6 if hl else 3),
        hovertemplate=f"pooled value {b.value:.2f} over {int(b.weight)} season(s)<extra></extra>"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Current fit (blocks)",
                         line=dict(color=COLORS["idr"], width=3)))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name="Block just pooled",
                         line=dict(color=COLORS["ink"], width=6)))
style(fig, "30 seasons, ordered from driest to wettest SEAS5 forecast", "Share ≤ t", height=340)
fig.update_yaxes(range=[-0.1, 1.1])
st.markdown(f"**Step {k} of {len(steps) - 1}.** {step.note}")
show(fig)
st.caption("The final step matches IDRModel fitted on the same 30 seasons, which is checked every time this page runs.")

# ---------------------------------------------------------------------------
st.divider()
st.markdown("#### The full answer for one forecast: the predictive CDF")
c1, c2 = st.columns([3, 1])
x_val = c1.slider("SEAS5 forecast for the season (mm)", 0, int(x_hi), int(np.median(s.x_train)), step=5,
                  key="lab2_x")
nb_view = c2.toggle("Compare 4 example forecasts", key="lab2_nb",
                    help="A dry, two middling and a wet SEAS5 value (10th, 40th, 70th and 90th percentiles of the training forecasts).")

lg = LinearGaussian.fit(s.x_train, s.y_train)
t_lo = min(0.0, lg.a + lg.b * x_val - 2.6 * lg.sigma)
t_axis = np.linspace(t_lo, y_hi * 1.15, 500)

fig = go.Figure()
if nb_view:
    example_x = np.quantile(s.x_train, [0.1, 0.4, 0.7, 0.9])
    for xx, col in zip(example_x, ORDINAL_BLUES):
        thr, cdf = predictive_cdf(model, xx)
        fig.add_trace(go.Scatter(x=np.r_[t_lo, thr], y=np.r_[0, cdf], mode="lines", line_shape="hv",
                                 name=f"SEAS5: {xx:.0f} mm", line=dict(color=col, width=2.5),
                                 hovertemplate="≤ %{x:.0f} mm: %{y:.2f}<extra>SEAS5 " + f"{xx:.0f}" + "</extra>"))
    fig.add_vline(x=t, line=dict(color=COLORS["ink2"], dash="dot", width=1),
                  annotation_text=f"t = {t} mm", annotation_position="top")
else:
    thr, cdf = predictive_cdf(model, x_val)
    q10, q50, q90 = predictive_quantiles(model, x_val, (0.1, 0.5, 0.9))
    fig.add_vrect(x0=q10, x1=q90, fillcolor=COLORS["idr"], opacity=0.08, line_width=0,
                  annotation_text="IDR 10–90% range", annotation_position="top left")
    fig.add_trace(go.Scatter(x=np.r_[t_lo, thr], y=np.r_[0, cdf], mode="lines", line_shape="hv",
                             name="IDR", line=dict(color=COLORS["idr"], width=2.5),
                             hovertemplate="P(CHIRPS ≤ %{x:.0f} mm) = %{y:.2f}<extra>IDR</extra>"))
    fig.add_trace(go.Scatter(x=t_axis, y=lg.cdf(x_val, t_axis), mode="lines", name="Linear + Gaussian",
                             line=dict(color=COLORS["compare"], width=2.5, dash="dash"),
                             hovertemplate="P(CHIRPS ≤ %{x:.0f} mm) = %{y:.2f}<extra>Linear+Gaussian</extra>"))
    f_t = float(cdf_at_rows(model.predict(np.array([x_val])), t)[0])
    fig.add_trace(go.Scatter(x=[t], y=[f_t], mode="markers", name=f"Height at t = {t} mm (from the plot above)",
                             marker=dict(size=11, color=COLORS["idr"], line=dict(color="white", width=2)),
                             hovertemplate="F(t | x) = %{y:.2f}<extra></extra>"))
    if t_lo < 0:
        fig.add_vrect(x0=t_lo, x1=0, fillcolor=COLORS["compare"], opacity=0.07, line_width=0,
                      annotation_text="negative rain!", annotation_position="bottom left")
style(fig, "Rainfall threshold (mm)", "P(CHIRPS ≤ threshold)", height=430)
fig.update_yaxes(range=[0, 1.02])
show(fig)

if not nb_view:
    g50 = lg.a + lg.b * x_val
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("IDR median", f"{q50:.0f} mm")
    m2.metric("IDR 10–90% range", f"{q10:.0f}–{q90:.0f} mm")
    m3.metric("Linear+Gaussian median", f"{g50:.0f} mm")
    m4.metric("Linear+Gaussian P(rain < 0)", f"{float(lg.cdf(x_val, 0.0)):.1%}",
              help="Probability given to physically impossible negative rainfall.")
explain("lab2.cdf.explain", "lab2.cdf.maths")

# ---------------------------------------------------------------------------
example_x = np.quantile(s.x_train, [0.1, 0.4, 0.7, 0.9])
probs = cdf_at_rows(model.predict(example_x), t)
table = "\n\n| SEAS5 forecast | P(CHIRPS ≤ " + f"{t} mm)" + " |\n|---|---|\n" + "\n".join(
    f"| {xx:.0f} mm | {pp:.2f} |" for xx, pp in zip(example_x, probs))
check_yourself("lab2.check", t_mm=t, example_table=table, x_dry=f"{example_x[0]:.0f}")
