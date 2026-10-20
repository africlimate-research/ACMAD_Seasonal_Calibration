# ACMAD OND IDR-only tutorial (offline)

**60 minutes | historical SEAS5 and CHIRPS OND rainfall (2001–2025)**

## Quick start

1. Extract the ZIP and keep the notebook, `idr_xarray.py` and `data/` in the same folder.
2. Install dependencies: `python -m pip install -r requirements.txt`.
3. Start Jupyter with `jupyter notebook` and open `ACMAD_OND_IDR_Offline_Practical.ipynb`.
4. Run all cells top to bottom. No internet, GCS credentials or local research directory required after installation.

## Included data

`data/seas5_OND.npy` and `data/chirps_obs_OND.npy`: matched historical SEAS5 predictors and CHIRPS observed seasonal totals, shape (25, 12424). `data/pred_years_OND.npy`: 2001–2025. `data/y_mean_wmo_OND.npy`, `y_std_wmo_OND.npy`: WMO-referenced local moments. `data/zone_mean_wmo_OND.npy`, `zone_std_wmo_OND.npy`: zone-derived reference moments, **not** climate-zone labels. `africa_standard_extent.json`: extent only, **not** a spatial index.

The tutorial uses 2001–2015 for model fitting, 2016–2025 as a held-out period, 1,200 training pairs and 1,500 evaluation pairs for speed. Pooled IDR and pooled terciles are classroom approximations, not a replication of research-grade zonal LOYO calibration. The saved input has historical ensemble-mean/predictor values rather than raw ensemble members; do not infer raw ensemble probabilistic skill.

## September 2026 extension

A September 2026 SEAS5 precipitation **NetCDF** can be used later **from a local path**; the file does not need to be uploaded. Section 5 contains an optional lazy `xarray.open_dataset(..., chunks={})` inspection cell that skips gracefully when the file is absent. Conversion from accumulated precipitation to OND totals and mapping of the 2026 forecast to the original 12,424 columns must be verified. The available data do not contain a spatial index; the bounding-box JSON alone cannot restore the mapping. The notebook deliberately does not invent OND 2026 probabilities.

### ROC discrimination extension

Section 4 includes one-versus-rest ROC curves and AUC scores for the below-normal, near-normal, and above-normal categories using the same 2016–2025 held-out samples as Brier skill and reliability. ROC AUC assesses ranking, not probability calibration. Install `scikit-learn` from requirements.txt.

## Local large NetCDF provision

In Section 5, edit `FORECAST_NC` to the absolute `.nc` path on your machine. Historical exercises run even with no 2026 file. To inspect large datasets lazily, install optional packages with `python -m pip install dask netcdf4`. Forecast aggregation, historic location alignment, forecast probability calculation and maps are **not automatically enabled** because the supplied 12,424-column historical data lack a confirmed pixel mapping and the NetCDF accumulation metadata are not yet known.

## Running on a shared JupyterHub

**Facilitator / admin (once):**

1. Install the packages into the hub's user image or shared kernel: `pip install numpy matplotlib xarray numba scikit-learn` (add `dask netcdf4` for Section 5). The `notebook` package is not needed on a hub.
2. Give each participant **their own copy** of the folder (notebook, `idr_xarray.py`, `data/`, `africa_standard_extent.json`). Upload the ZIP, use nbgitpuller from a Git repo, or copy from a read-only shared folder: `cp -r /path/to/shared/ACMAD_IDR_2026 ~/`. Do not ship `__pycache__/`; it contains machine-specific Numba caches.
3. If the September 2026 NetCDF is provided, put it in one shared read-only location. Participants can then set `FORECAST_NC` in Section 5 or the environment variable `SEAS5_2026_NC`. Users should not copy the file.

**Participants:** open the notebook from inside your copy of the folder and run **Section 0 — Setup** first. It limits each kernel to 2 CPU threads (override with env var `IDR_THREADS`), keeps the Numba cache in `~/.cache/numba`, and reports any missing packages. If packages are missing and the admin cannot install them, run `%pip install --user <packages>` in a cell and restart the kernel.

Resources: each kernel uses well under 1 GB RAM. The first IDR fit takes a few seconds while Numba compiles.

## Running on Google Colab

Each participant runs their own Colab session, so there are no shared threads or folders to configure. Colab already includes the required packages; Section 0 installs any that are missing.

**Facilitator (once):**

1. Host `ACMAD_IDR_2026_hub.zip` at a public direct-download link, for example a GitHub repo (`https://github.com/ORG/REPO/raw/main/ACMAD_IDR_2026_hub.zip`) or Google Drive shared as "Anyone with the link" (`https://drive.google.com/uc?export=download&id=FILE_ID`).
2. Paste that link into `TUTORIAL_ZIP_URL` in the Section 0 cell of the notebook.
3. Share the notebook. Either put it on Google Drive and share it as Viewer (participants choose **Open with → Google Colaboratory**), or push it to GitHub and share `https://colab.research.google.com/github/ORG/REPO/blob/main/ACMAD_OND_IDR_Offline_Practical.ipynb`.

**Participants:** open the link, choose **File → Save a copy in Drive**, then **Runtime → Run all**. If `TUTORIAL_ZIP_URL` is empty, Section 0 asks you to upload the ZIP. Colab deletes uploaded files when the session ends, so run Section 0 again after reconnecting.

**2026 NetCDF on Colab:** the file is large, so upload it to Google Drive once. Participants add a shortcut to it in their own Drive and mount Drive (see the comment in Section 5).
