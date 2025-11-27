# EGMS Ortho Product Pipeline

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/FilSantarelli/egms-ortho-product/master?urlpath=%2Fdoc%2Ftree%2Fscript.ipynb)

## Purpose
The goal of this repository is to reproduce European Ground Motion Service (EGMS) L3 ORTHO products from Sentinel-1 L2a/L2b inputs. The included `script.ipynb` notebook orchestrates the full workflow: it standardises burst acquisitions, builds a regular spatio-temporal grid, solves per-cell least-squares (LS) systems that combine ascending/descending geometries, and formats the resulting east/vertical displacement products according to EGMS naming conventions.

## High-level algorithm
1. **Burst ingestion** – Load CSV or zipped CSV bursts through the shared `egms_io` helpers, tagging each row with its source for traceability.
2. **Temporal normalisation** – Interpolate acquisition columns so every burst shares a common cadence, filling gaps up to a configurable length.
3. **Rasterisation** – Encode each observation into a 64-bit cell identifier that packs easting, northing, and time indices anchored to the AOI/grid definition, to exploit the full potential of pandas's groupby operations (the algorithm scales up easily with Dask clusters)
4. **Least-squares solving** – Group by cell ID and solve separate LS problems for displacements (east/up) and GNSS velocities (east/up/north), discarding under-determined cells.
5. **Cube reconstruction** – Decode cell IDs back to coordinates, pivot timelines into GeoDataFrames, and attach static metadata (height, GNSS terms).
6. **Model estimation & export** – Finally estimate the model to compute displacement statistics, compute product IDs, format fields according to EGMS specifications, and write ZIP/Parquet outputs.

## Repository layout
- `script.ipynb` – interactive end-to-end pipeline that mirrors the algorithm above.
- `environment.yml` – conda specification for a reproducible environment (Python 3.13, Dask, GeoPandas, Rasterio, etc.).
- `egms_io.py`, `format_dataframe.py`, `metadata.py`, `model_estimation.py`, `model.py` – reusable helpers for I/O, formatting, metadata derivation, and physical model calculations.
- `base_encoding.py` – utilities for mapping between grid indices and encoded identifiers.
- `test/` – sample inputs/outputs used to validate the workflow.
- `LICENSE` – project licensing terms.

## Running the notebook
### Option 1: Binder (no local setup)
Click the Binder badge above. Binder will build the conda environment defined in `environment.yml` and open `script.ipynb` directly in JupyterLab. Upload or mount your Sentinel-1 bursts into the session storage, adjust the "Parameters" cell, and execute the notebook sequentially.

Binder resource limits (per [mybinder.org usage guidelines](https://mybinder.readthedocs.io/en/latest/about/user-guidelines.html)):
- You are guaranteed 1 GB RAM and may burst up to 2 GB; exceeding 2 GB restarts the kernel.
- Sessions are culled after ~10 minutes of inactivity and generally run up to 6 hours (or about one CPU-hour for heavy jobs).
- Storage inside Binder is entirely ephemeral—any files saved during the session disappear once it stops, so download results before closing.

### Option 2: Local execution
1. Clone the repository and move into the project folder:
   ```bash
   git clone https://github.com/FilSantarelli/egms-ortho-product.git
   cd egms-ortho-product/script-ortho
   ```
2. Create the conda environment and activate it:
   ```bash
   conda env create -f environment.yml
   conda activate egms-ortho
   ```
3. Launch JupyterLab (or Notebook) and open `script.ipynb`:
   ```bash
   jupyter lab
   ```
4. Update the parameter cell to point to your burst set, AOI polygon, temporal window, and output directory.
5. Run cells in order. The notebook spins up a local Dask `LocalCluster`; adjust worker counts/memory as needed for your hardware or replace it with your cluster scheduler of choice.

### Data expectations
- Sentinel-1 L2a/L2b bursts stored as CSV/ZIP in `test/<AOI>/input/` (default path). Update the glob if your files live elsewhere.
- AOI geometry is defined as an EPSG:3035-aligned box; update `aoi` if you require a more complex polygon.
- Outputs are written to `output/` by default, including EGMS-compliant ZIP archives and Parquet copies for a fast check.

## Notebook workflow tips
- **Monitoring** – The Dask dashboard URL is printed when the cluster starts; open it in a browser to monitor task graphs, memory, and worker health.
- **Partition sizing** – The rasterised Dask DataFrame is repartitioned to ~4 MB chunks to keep LS solves memory-friendly; tweak `repartition` if your workload deviates.
- **Quality control** – Intermediate GeoDataFrames retain a `path` column so you can trace anomalies back to specific bursts before exporting.

## Output
The final ZIP packages comply with EGMS L3 naming conventions, embedding PID, easting/northing, orthometric height, velocity/acceleration statistics, and GNSS-derived components. Companion Parquet files mirror the same schema for rapid inspection.
