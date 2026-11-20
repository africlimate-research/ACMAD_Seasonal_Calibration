import streamlit as st

from app.ui import get_ds, pooled, settings, text

st.title("Seasonal rainfall calibration with IDR")
st.caption("An interactive companion to the ACMAD OND practical: SEAS5 → CHIRPS, 2001–2025")

st.markdown(text("home.intro"))
st.info(text("home.pipeline"), icon=":material/route:")

ds, r = get_ds(), pooled()
cfg = settings()
st.markdown("#### The setup")
st.markdown(text("home.setup", n_years=ds.n_years, y0=ds.years[0], y1=ds.years[-1], n_loc=f"{ds.n_locations:,}",
                 split=cfg.split_year, split1=cfg.split_year + 1, n_train=f"{cfg.n_train:,}",
                 n_test=f"{cfg.n_test:,}", seed=cfg.seed, n_test_years=r.sample.n_test_years))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Seasons", ds.n_years)
c2.metric("Locations", f"{ds.n_locations:,}")
c3.metric("Training pairs used", f"{len(r.sample.x_train):,}")
c4.metric("Held-out pairs scored", f"{len(r.sample.x_test):,}")

st.markdown("#### The learning path")
steps = [("1", "Meet the data", "SEAS5 vs CHIRPS: what the forecast gets right, and what it mostly knows (geography)."),
         ("2", "How IDR works", "One number in, a full set of probabilities out, built one threshold at a time."),
         ("3", "From curve to terciles", "Reading P(below), P(near) and P(above normal) off the curve."),
         ("4", "Is it any good?", "Brier score, reliability and ROC: what each one sees and misses."),
         ("5", "Traps and illusions", "Why the headline skill is mostly geography, and other ways scores mislead."),
         ("6", "Question bank", "Every question from the practical, each linked to the plot that answers it.")]
cols = st.columns(3)
files = ["lab1_data", "lab2_idr", "lab3_terciles", "lab4_scores", "lab5_traps", "lab6_questions"]
for i, ((n, title, blurb), f) in enumerate(zip(steps, files)):
    with cols[i % 3].container(border=True, height="stretch"):
        st.page_link(f"labs/{f}.py", label=f"**Lab {n} · {title}**", icon=":material/arrow_forward:")
        st.caption(blurb)
st.markdown("New words? The **Glossary** (bottom of the sidebar menu) explains every term in plain language.")
st.caption("Source: the ACMAD OND IDR practical notebook (`ACMAD_OND_IDR_Offline_Practical.ipynb`). "
           "Each lab names the notebook section it follows, for anyone who wants to read along.")
with st.expander("What this data is *not*"):
    st.markdown(
        "- **Not ensemble members.** SEAS5 is one predictor value per season and location, so raw-ensemble skill can't be computed.\n"
        "- **No coordinates.** We don't know where each of the 12,424 locations is, so there are **no maps**.\n"
        "- **The zone arrays are not zone labels**, so zone-by-zone IDR can't be recreated."
    )
