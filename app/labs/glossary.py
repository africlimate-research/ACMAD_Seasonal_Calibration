import streamlit as st

from app.ui import text

st.title("Glossary")
st.caption("Every term used in the labs, in plain words")
q = st.text_input("Filter", placeholder="Type a word, e.g. tercile", key="gloss_q").strip().lower()
entries = [e for e in text("glossary").split("\n\n") if e.strip()]
shown = [e for e in entries if q in e.lower()] if q else entries
for e in shown:
    st.markdown(e)
if not shown:
    st.info("No term matches that filter.", icon=":material/search_off:")
