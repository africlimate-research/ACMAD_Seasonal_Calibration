"""Phase 1 parity tests. Run from the repo root:  .venv/bin/python -m pytest app/tests -q"""
import numpy as np
import pytest

from app.core.data import load_data
from app.core.experiment import (IDRModel, LinearGaussian, Settings, cdf_at_rows, categories, draw_sample,
                                 run_pooled, shuffle_experiment, threshold_curve)
from app.core.location import pooled_r, per_location_r, pooling_comparison
from app.core.scores import bss, brier_table, confusion, murphy, reliability, roc, year_bootstrap_bss


@pytest.fixture(scope="module")
def ds():
    return load_data()


@pytest.fixture(scope="module")
def nb(ds):
    """Notebook cells 9, 13, 16, 18 copied verbatim (independent reference implementation)."""
    X, Y, YEARS = ds.X, ds.Y, ds.years
    is_train = YEARS <= 2015
    is_test = YEARS >= 2016
    x_train_all, y_train_all = X[is_train].ravel(), Y[is_train].ravel()
    x_test_all, y_test_all = X[is_test].ravel(), Y[is_test].ravel()
    seed = np.random.default_rng(42)
    i_train = seed.choice(len(x_train_all), size=min(1200, len(x_train_all)), replace=False)
    i_test = seed.choice(len(x_test_all), size=min(1500, len(x_test_all)), replace=False)
    xtrain, ytrain = x_train_all[i_train], y_train_all[i_train]
    xtest, ytest = x_test_all[i_test], y_test_all[i_test]
    idr = IDRModel().fit(xtrain, ytrain)
    q1, q2 = np.quantile(y_train_all, [1 / 3, 2 / 3])
    pred = idr.predict(xtest)
    p_bn = np.atleast_1d(pred.cdf_at(q1))
    p_not_an = np.atleast_1d(pred.cdf_at(q2))
    P = np.column_stack([p_bn, p_not_an - p_bn, 1 - p_not_an])
    obs_cat = np.digitize(ytest, [q1, q2], right=True)
    train_cat = np.digitize(y_train_all, [q1, q2], right=True)
    p_clim = np.array([(train_cat == j).mean() for j in range(3)])
    bss_nb = []
    for j in range(3):
        outcome = (obs_cat == j).astype(float)
        bss_nb.append(1 - np.mean((P[:, j] - outcome) ** 2) / np.mean((p_clim[j] - outcome) ** 2))
    from sklearn.metrics import roc_auc_score
    aucs = [roc_auc_score((obs_cat == j).astype(int), P[:, j]) for j in range(3)]
    return dict(xtrain=xtrain, xtest=xtest, q1=q1, q2=q2, P=P, obs_cat=obs_cat, p_clim=p_clim,
                bss=np.array(bss_nb), auc=np.array(aucs))


@pytest.fixture(scope="module")
def res(ds):
    return run_pooled(ds, Settings())


# --- Notebook parity ---------------------------------------------------------

def test_default_settings_flag():
    assert Settings().is_notebook_default
    assert not Settings(seed=1).is_notebook_default


def test_sample_matches_notebook(res, nb):
    np.testing.assert_array_equal(res.sample.x_train, nb["xtrain"])
    np.testing.assert_array_equal(res.sample.x_test, nb["xtest"])


def test_sample_keeps_location_and_year(ds, res):
    s = res.sample
    rows = {int(y): i for i, y in enumerate(ds.years)}
    for k in range(0, 1500, 97):
        assert ds.X[rows[int(s.year_test[k])], s.loc_test[k]] == s.x_test[k]
        assert ds.Y[rows[int(s.year_test[k])], s.loc_test[k]] == s.y_test[k]
    assert s.year_train.max() <= 2015 < s.year_test.min()
    assert s.n_test_years == 10


def test_terciles_probabilities_and_categories(res, nb):
    assert res.q1 == pytest.approx(nb["q1"]) and res.q2 == pytest.approx(nb["q2"])
    np.testing.assert_allclose(res.P, nb["P"], atol=1e-12)
    np.testing.assert_array_equal(res.obs_cat, nb["obs_cat"])
    np.testing.assert_allclose(res.clim, nb["p_clim"])
    assert np.all(res.P >= -1e-8) and np.allclose(res.P.sum(1), 1, atol=1e-7)


def test_bss_parity(res, nb):
    got = bss(res.P, res.obs_cat, res.clim)
    np.testing.assert_allclose(got, nb["bss"], atol=1e-12)
    np.testing.assert_allclose(got, [0.723, 0.517, 0.774], atol=5e-4)
    table = brier_table(res.P, res.obs_cat, res.clim)
    np.testing.assert_allclose([r["bss"] for r in table], got)


def test_roc_auc_parity(res, nb):
    aucs = [roc(res.P[:, j], res.obs_cat == j).auc for j in range(3)]
    np.testing.assert_allclose(aucs, nb["auc"], atol=1e-12)


# --- Plan §2 numbers -----------------------------------------------------------

def test_correlations(ds):
    assert pooled_r(ds) == pytest.approx(0.935, abs=1e-3)
    assert pooled_r(ds, anomalies=True) == pytest.approx(0.32, abs=5e-3)
    assert np.nanmedian(per_location_r(ds)) == pytest.approx(0.17, abs=5e-3)


def test_pooling_comparison_review_numbers(ds):
    c = pooling_comparison(ds, common=False)
    np.testing.assert_allclose(c.pooled, [0.723, 0.517, 0.774], atol=5e-4)
    np.testing.assert_allclose(c.pooled_local, [-0.212, -0.044, -0.165], atol=5e-4)
    np.testing.assert_allclose(c.anomaly_local, [0.027, 0.023, 0.047], atol=5e-4)
    assert c.n_locations_equal_cutoffs == 44 and c.n_test_excluded == 6
    assert c.n_train_pairs_anomaly == 1020


def test_pooling_comparison_common_pairs(ds):
    c = pooling_comparison(ds, common=True)
    assert c.n_test_pairs == 1296
    assert c.pooled[0] > 0.5 and c.pooled_local[0] < 0 and 0 < c.anomaly_local[2] < 0.1


def test_shuffle_experiment(ds):
    r = shuffle_experiment(ds, n_shuffles=100)
    np.testing.assert_allclose(r["shuffled"].mean(0), [0.042, -0.003, 0.045], atol=5e-4)
    np.testing.assert_allclose(r["wrong_direction"].mean(0), [-0.073, -0.005, -0.064], atol=5e-4)
    np.testing.assert_allclose(r["shuffled_train_and_test"].mean(0), [-0.011, -0.002, -0.009], atol=5e-4)
    np.testing.assert_allclose(r["constant_climatology"][0], [-0.008, 0.0, -0.005], atol=5e-4)


# --- Helpers used by the labs ------------------------------------------------

def test_threshold_curve_is_monotone_and_matches_event(res):
    t = res.q1
    xs, p = threshold_curve(res.model, res.sample.x_train, t)
    assert np.all(np.diff(xs) > 0)
    assert np.all(np.diff(p) <= 1e-9), "P(Y<=t|x) must be non-increasing in x"
    # Exactly the event plotted: evaluating at t must equal cdf_at(t), not a snapped threshold.
    np.testing.assert_allclose(p, np.atleast_1d(res.model.predict(xs).cdf_at(t)))


def test_cdf_at_rows_scalar_and_vector(res):
    d = res.model.predict(res.sample.x_test[:5])
    np.testing.assert_allclose(cdf_at_rows(d, res.q1), d.cdf_at(res.q1))
    t = np.array([res.q1, res.q2, res.q1, res.q2, res.q1])
    np.testing.assert_allclose(cdf_at_rows(d, t), [d.cdf_at(t[i])[i] for i in range(5)])


def test_categories_match_digitize():
    y = np.array([0, 1, 1.5, 2, 3])
    np.testing.assert_array_equal(categories(y, 1, 2), np.digitize(y, [1, 2], right=True))


def test_murphy_adds_up(res):
    for j in range(3):
        p, o = res.P[:, j], (res.obs_cat == j).astype(float)
        m = murphy(p, o, n_bins=8)
        assert m.bs_binned == pytest.approx(m.reliability - m.resolution + m.uncertainty, abs=1e-12)
        assert m.bs_raw == pytest.approx(m.bs_binned + m.within_bin)


def test_reliability_counts_all_forecasts(res):
    r = reliability(res.P[:, 2], (res.obs_cat == 2).astype(float), n_bins=8)
    assert r.count.sum() == len(res.P)


def test_confusion_totals(res):
    c = confusion(res.P[:, 0], res.obs_cat == 0, 0.5)
    assert sum(c.values()) == len(res.P)


def test_year_bootstrap_shape(res):
    b = year_bootstrap_bss(res.P, res.obs_cat, res.clim, res.sample.year_test, n_boot=20)
    assert b.shape == (20, 3) and np.isfinite(b).all()


def test_linear_gaussian_cdf_is_valid(res):
    lg = LinearGaussian.fit(res.sample.x_train, res.sample.y_train)
    c = lg.cdf(300.0, np.linspace(-500, 3000, 50))
    assert np.all(np.diff(c) >= 0) and c[0] < 0.01 and c[-1] > 0.99


def test_non_default_settings_run(ds):
    r = run_pooled(ds, Settings(seed=7, n_train=500, n_test=400, split_year=2012))
    assert r.P.shape == (400, 3) and r.sample.year_train.max() <= 2012
    assert draw_sample(ds, Settings(seed=7)).x_train.shape == (1200,)


# --- Display-only PAVA must agree with IDRModel ------------------------------

def test_display_pava_matches_idr(res):
    from app.core.experiment import fit_idr
    from app.core.pava import expand, group_by_x, pava_steps_antitonic

    s = res.sample
    order = np.argsort(s.x_train)
    pick = order[np.linspace(0, len(order) - 1, 30).astype(int)]
    x, y = s.x_train[pick], s.y_train[pick]
    model = fit_idr(x, y)
    for t in np.quantile(y, [0.2, 0.5, 0.8]):
        ux, means, w = group_by_x(x, y <= t)
        steps = pava_steps_antitonic(means, w)
        final = expand(steps[-1], len(ux))
        assert np.all(np.diff(final) <= 1e-12)
        np.testing.assert_allclose(final, threshold_curve(model, x, t)[1], atol=1e-3)  # predict rounds to 3 dp


# --- Lab 1 / 4D / 5 helpers (app/core/traps.py) ------------------------------------------

def test_leakage_honest_matches_notebook_and_trap_a(ds, nb):
    from app.core.traps import leakage_experiment
    np.testing.assert_allclose(leakage_experiment(ds).bss, nb["bss"], atol=1e-12)
    loc = leakage_experiment(ds, local=True)
    pc = pooling_comparison(ds, common=True)
    assert loc.n_test == pc.n_test_pairs == 1296
    np.testing.assert_allclose(loc.bss, pc.anomaly_local, atol=1e-12)


def test_leakage_directions(ds):
    from app.core.traps import leakage_experiment
    honest = leakage_experiment(ds, local=True).bss
    assert (leakage_experiment(ds, local=True, leak_reference=True).bss < honest - 0.05).all()   # yardstick too good
    assert (leakage_experiment(ds, local=True, leak_model=True).bss > honest).all()              # model saw answers


def test_bias_by_amount(ds):
    from app.core.traps import bias_by_amount
    b = bias_by_amount(ds)
    assert len(b.mean_bias) == 10 and b.mean_bias[-1] > 50 and b.mean_bias[3] < 0
    assert b.overall_mean == pytest.approx(float((ds.X - ds.Y).mean()), rel=1e-5)


def test_seed_sensitivity_and_bootstraps(ds, res):
    from app.core.traps import bss_by_year, pair_bootstrap_bss, seed_sensitivity
    sd = seed_sensitivity(ds, n_seeds=3)
    assert sd.shape == (3, 3) and np.isfinite(sd).all()
    pb = pair_bootstrap_bss(res.P, res.obs_cat, res.clim, n_boot=50)
    assert pb.shape == (50, 3)
    uy, v, n = bss_by_year(res.P, res.obs_cat, res.clim, res.sample.year_test)
    assert list(uy) == list(range(2016, 2026)) and n.sum() == len(res.obs_cat) and v.shape == (10, 3)


def test_installed_versions_meet_declared_minimums():
    """Compatibility: the tested environment must satisfy app/requirements.txt (Codex follow-up review)."""
    import re
    from importlib.metadata import version
    from pathlib import Path
    from packaging.specifiers import SpecifierSet
    req = Path(__file__).resolve().parents[1] / "requirements.txt"
    for line in req.read_text().splitlines():
        m = re.match(r"^(streamlit|plotly)([<>=!,.\d]+)$", line.strip())
        if m:
            assert version(m.group(1)) in SpecifierSet(m.group(2)), f"{m.group(1)} {version(m.group(1))} vs {m.group(2)}"
