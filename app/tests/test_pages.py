"""Smoke-test every page headlessly. Run: .venv/bin/python -m pytest app/tests -q"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

PAGES = {"labs/home.py": ("Seasonal rainfall calibration with IDR", 0),
         "labs/lab1_data.py": ("Lab 1 · Meet the data", 4),
         "labs/lab2_idr.py": ("Lab 2 · How IDR works", 3),
         "labs/lab3_terciles.py": ("Lab 3 · From curve to terciles", 3),
         "labs/lab4_scores.py": ("Lab 4 · Is the forecast any good?", 6),
         "labs/lab5_traps.py": ("Lab 5 · Traps and illusions", 4),
         "labs/lab6_questions.py": ("Lab 6 · Question bank", 0),
         "labs/glossary.py": ("Glossary", 0)}


def _run(page, setup=None):
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=120)
    at.run()
    if page != "labs/home.py":
        at.switch_page(page)
        at.run()
    if setup:
        setup(at)
        at.run()
    assert not at.exception, [e.value for e in at.exception]
    title, n_charts = PAGES[page]
    assert [t.value for t in at.title] == [title], "page did not render"
    assert len(at.get("plotly_chart")) >= n_charts
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page):
    at = _run(page)
    assert "missing text" not in " ".join(m.value for m in at.markdown)


def test_lab2_extreme_threshold_and_log():
    def setup(at):
        at.slider(key="lab2_t").set_value(0)
        at.toggle(key="lab2_log").set_value(True)
        at.toggle(key="lab2_nb").set_value(True)
    _run("labs/lab2_idr.py", setup)


@pytest.mark.parametrize("loc", ["Dry place (10th percentile)", "Wet place (90th percentile)"])
@pytest.mark.parametrize("mode", ["Pooled (one pair for all of Africa)", "Local (this location)"])
def test_lab3_modes(loc, mode):
    def setup(at):
        at.radio(key="lab3_mode").set_value(mode)
        at.selectbox(key="lab3_loc").set_value(loc)
    _run("labs/lab3_terciles.py", setup)


def test_lab3_equal_cutoff_location_falls_back():
    import numpy as np
    from app.core.data import load_data
    from app.core.location import local_terciles
    ds = load_data()
    bad = int(np.flatnonzero(~local_terciles(ds, ds.years <= 2015).distinct)[0])

    def setup(at):
        at.radio(key="lab3_mode").set_value("Local (this location)")
        at.selectbox(key="lab3_loc").set_value("Choose by column index")
        at.run()
        at.number_input(key="lab3_idx").set_value(bad)
    at = _run("labs/lab3_terciles.py", setup)
    assert any("undefined" in w.value for w in at.warning)


def test_custom_settings_flag():
    def setup(at):
        at.number_input(key="set_seed").set_value(7)
    at = _run("labs/home.py", setup)
    assert any("Custom settings" in w.value for w in at.warning)


def test_home_setup_filled_and_tracks_settings():
    at = _run("labs/home.py")
    md = " ".join(m.value for m in at.markdown)
    assert "{" not in md and "2001–2015" in md and "1,200 random" in md
    at.slider(key="set_split").set_value(2012)
    at.run()
    md = " ".join(m.value for m in at.markdown)
    assert "2013–2025" in md


def test_lab1_anomalies_and_wet_place():
    def setup(at):
        at.radio(key="lab1_view").set_value("Anomalies: each location's own average removed")
        at.selectbox(key="lab1_loc").set_value("Wet place (90th percentile)")
    at = _run("labs/lab1_data.py", setup)
    assert any(m.value == "0.32" for m in at.metric)


@pytest.mark.parametrize("cat", ["BN", "NN"])
def test_lab4_controls(cat):
    def setup(at):
        at.radio(key="lab4_cat").set_value(cat)
        at.slider(key="lab4_bins").set_value(15)
        at.slider(key="lab4_cut").set_value(0.05)
    _run("labs/lab4_scores.py", setup)


@pytest.mark.parametrize("mode", ["Pooled (the notebook)", "Local (anomaly model, own cutoffs)"])
def test_lab5_traps_switches(mode):
    def setup(at):
        at.radio(key="lab5_a").set_value("③ Location-aware (anomaly) model, own cutoffs")
        at.radio(key="lab5_b_mode").set_value(mode)
        at.toggle(key="lab5_b_ref").set_value(True)
        at.toggle(key="lab5_b_model").set_value(True)
        at.toggle(key="lab5_c_run").set_value(True)
        at.radio(key="lab5_e_mode").set_value("Location-aware (anomaly model, own cutoffs)")
    at = _run("labs/lab5_traps.py", setup)
    assert len(at.get("plotly_chart")) >= 6     # shuffle chart appears once run


def test_question_bank_link_presets_target_page():
    at = _run("labs/lab6_questions.py")
    at.button(key="go_dry_threshold_0").click()
    at.run()
    assert not at.exception
    assert [t.value for t in at.title] == ["Lab 2 · How IDR works"]
    assert at.toggle(key="lab2_nb").value is True


def test_glossary_filter():
    def setup(at):
        at.text_input(key="gloss_q").input("tercile")
    at = _run("labs/glossary.py", setup)
    md = " ".join(m.value for m in at.markdown)
    assert "Tercile" in md and "SEAS5**" not in md


# --- Fixes from the implementation review --------------------------------------------------

def test_answers_survive_following_a_link_and_coming_back():
    at = _run("labs/lab6_questions.py")
    at.text_area(key="qb_answer_bias_amount").input("My draft")
    at.button(key="go_bias_amount_0").click()
    at.run()
    assert [t.value for t in at.title] == ["Lab 1 · Meet the data"]
    at.text_area(key="answer_lab1.check").input("Lab 1 draft")
    at.run()
    at.switch_page("labs/lab6_questions.py")
    at.run()
    assert at.text_area(key="qb_answer_bias_amount").value == "My draft"
    at.switch_page("labs/lab1_data.py")
    at.run()
    assert at.text_area(key="answer_lab1.check").value == "Lab 1 draft"


def test_explanations_follow_the_sign_of_the_live_scores():
    def setup(at):
        at.number_input(key="set_seed").set_value(0)
        at.slider(key="set_ntrain").set_value(200)
        at.slider(key="set_ntest").set_value(200)
        at.slider(key="set_split").set_value(2020)
        at.run()
        at.radio(key="lab5_a").set_value("③ Location-aware (anomaly) model, own cutoffs")
    at = _run("labs/lab5_traps.py", setup)
    md = " ".join(m.value for m in at.markdown)
    assert "it is worse than climatology (BSS -0.25)" in md
    assert "beats local climatology" not in md


def test_question_bank_link_selects_lab5_tab():
    at = _run("labs/lab6_questions.py")
    at.button(key="go_reliable_no_skill_0").click()
    at.run()
    assert [t.value for t in at.title] == ["Lab 5 · Traps and illusions"]
    assert at.session_state["lab5_tab"] == "D · Reliable ≠ skilful"


def test_question_bank_does_not_run_shuffles_on_entry():
    at = _run("labs/lab6_questions.py")
    assert "_shuffle_done" not in at.session_state
    at.button(key="qb_run_shuffle").click()
    at.run()
    assert "_shuffle_done" in at.session_state
    md = " ".join(m.value for m in at.markdown)
    assert "collapses from" in md
