# ACMAD OND IDR practical

**60-minute tutorial | Historical SEAS5 and CHIRPS OND rainfall (2001–2025)**

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/africlimate-research/ACMAD_Seasonal_Calibration/blob/main/ACMAD_OND_IDR_Offline_Practical.ipynb)

## Run the notebook

### Google Colab

Click **Open in Google Colab** above. Run **Section 0 — Setup** first; it clones this repository and installs the required packages. Then run the remaining cells from top to bottom. Colab clears its files when the runtime disconnects, so run Setup again after reconnecting.

### Run locally

Keep the notebook, `idr_xarray.py`, `africa_standard_extent.json`, and `data/` together. Install the dependencies and start Jupyter:

```bash
python -m pip install -r data/requirements.txt
jupyter notebook
```

Open `ACMAD_OND_IDR_Offline_Practical.ipynb` and run the cells from top to bottom. Historical exercises do not require internet access after dependencies are installed.

## Interactive learning app

A local Streamlit app that turns the practical into self-contained interactive labs: the data (Lab 1), how IDR works (Lab 2), terciles (Lab 3), verification scores (Lab 4), traps and illusions (Lab 5), a question bank (Lab 6) and a glossary. From the repository root:

```bash
python -m venv .venv
.venv/bin/pip install -r app/requirements.txt
.venv/bin/streamlit run app/app.py          # opens http://localhost:8501
.venv/bin/python -m pytest app/tests -q     # notebook-parity and page tests
```

Explanatory text lives in [`app/content/questions.md`](app/content/questions.md) and can be edited without touching code.

## Companion notebook: check the app yourself

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/africlimate-research/ACMAD_Seasonal_Calibration/blob/main/ACMAD_OND_IDR_App_Companion.ipynb)

[`ACMAD_OND_IDR_App_Companion.ipynb`](ACMAD_OND_IDR_App_Companion.ipynb) recomputes the app's numbers and plots, lab by lab, with NumPy, Matplotlib, scikit-learn and `idr_xarray` only. It imports none of the app's code. Each section names the app page it checks, and ✓ lines show a number exactly as the app displays it at the default settings (113 checks). It needs the same packages as the original notebook and runs in about a minute. Section 6 runs the Lab 5 experiments, which are not in the original notebook. `app/tests/test_companion_notebook.py` runs it and fails if any check stops matching the app.

## Included data and interpretation

`data/seas5_OND.npy` and `data/chirps_obs_OND.npy` contain matched historical SEAS5 predictors and CHIRPS observed seasonal totals (shape `(25, 12424)`). `data/pred_years_OND.npy` contains 2001–2025. The `y_mean_wmo_OND.npy` and `y_std_wmo_OND.npy` files contain WMO-referenced local moments; `zone_mean_wmo_OND.npy` and `zone_std_wmo_OND.npy` contain zone-derived reference moments, **not** climate-zone labels. `africa_standard_extent.json` provides an extent only, **not** a spatial index.

The tutorial fits on 2001–2015 and evaluates on the held-out 2016–2025 period, using 1,200 training pairs and 1,500 evaluation pairs for speed. Pooled IDR and pooled terciles are classroom approximations, not research-grade zonal leave-one-year-out calibration. The saved SEAS5 input contains historical predictor values, not raw ensemble members; raw-ensemble probabilistic skill cannot be inferred from these files.

## Optional September 2026 NetCDF extension

Section 5 can inspect a September 2026 SEAS5 precipitation NetCDF from a local path. Set `FORECAST_NC` to the file path; the cell skips gracefully if the file is absent. Install `dask` and `netcdf4` to inspect large datasets lazily:

```bash
python -m pip install dask netcdf4
```

Forecast aggregation, location alignment, probabilities, and maps are not enabled: the historical files do not include a confirmed pixel mapping, and the NetCDF accumulation metadata must be checked first. The notebook therefore does not generate OND 2026 probabilities.

## Requirements

The required packages are listed in [`data/requirements.txt`](data/requirements.txt). For a shared JupyterHub, an administrator can install them in the shared environment with:

```bash
python -m pip install numpy matplotlib xarray numba scikit-learn
```
