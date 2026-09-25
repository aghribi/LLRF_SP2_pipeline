# EPICS Channel Archiver reader

A small utility to read PV (process variable) history out of an EPICS
["Channel Archiver"](https://epics.anl.gov/extensions/archiver/) MySQL backend — the
kind of slow-control archive many accelerator facilities run alongside their EPICS
control system.

## Credit

Adapted from Charly Lassalle's `GetEPICSdata` package, written for his SPIRAL2
heat-load-observer PhD thesis work at GANIL:
https://gitlab.in2p3.fr/charly-lassalle-phd-thesis/programmes (MIT License, Copyright
(c) 2025 Charly Lassalle). This version keeps only the core archiver-read function,
translated to English, with parameterized SQL (the original built queries by string
concatenation) and a pandas `DataFrame` return instead of an `astropy.table.Table`, to
keep the dependency footprint small. See `LICENSE` in this folder for the original
license text.

Used in this project to pull GANIL slow-control telemetry (cryogenics, vacuum,
BPM/BLM, RF) as candidate precursor signals ahead of LLRF faults — see the
[Precursor Root-Cause Atlas](https://aghribi.github.io/LLRF_SP2_pipeline/precursor-dashboard.html).

## What it does

`read_pv_history(config, pv_name, time_start, time_end)` returns every sample recorded
for one PV in a given time window, as a two-column DataFrame (`time`, `value`).

The schema it queries — a `channel` table mapping PV names to IDs, and a `sample`
table of `(channel_id, smpl_time, nanosecs, float_val)` rows — is the standard EPICS
Channel Archiver RDB layout, so this should work against any facility's archiver built
on that same tool, not just GANIL's.

## Data samples

`data_samples/` has the full GANIL SPIRAL2 EPICS PV inventory this project surveyed
(4,497 PVs, categorized by subsystem) and the per-cryomodule extraction list built
from it — real `pv_name` values usable as-is with `read_pv_history` against a
compatible archiver, or as a reference for what a PV inventory for another
facility's archiver might look like. See `data_samples/README.md` for details.

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Copy `config.example.yaml` to `config.yaml` and fill in your archiver's real
   connection details. **`config.yaml` is gitignored — never commit real
   credentials.**

## Usage

```python
import yaml
from epics_archiver_reader import read_pv_history

with open("config.yaml") as f:
    config = yaml.safe_load(f)

df = read_pv_history(
    config,
    pv_name="CMA01-CRYO:TT001:TempMes",
    time_start="2026-01-01 00:00:00",
    time_end="2026-01-01 04:00:00",
)
print(df.head())
```

See `example.py` for a runnable version of the above.

## Notes

- Timestamps are passed straight through to MySQL for the `smpl_time` comparison —
  use whatever string format your server accepts for a `DATETIME`.
- An empty result means either the PV name has no matching row in `channel`, or the
  archiver has no samples for it in that window (e.g. the window predates the
  archiver's retention).
- This reads one PV per call. For many PVs across many time windows, loop over
  `read_pv_history` — the original thesis package also includes a Dash-based
  interactive browser built on top of the same query, not reproduced here to keep
  this utility dependency-light.
