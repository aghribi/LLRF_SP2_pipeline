# Data samples: GANIL SPIRAL2 EPICS PV inventory

Full inventory of process variables (PVs) recorded by GANIL's SPIRAL2 slow-control
EPICS archiver, surveyed 2026-09-09 and used as input to
[`epics_archiver_reader.py`](../epics_archiver_reader.py) (i.e. these are the `pv_name`
values you'd pass to `read_pv_history`).

Source: 10 EPICS "Channel Archiver" engines on the SPIRAL2 control network,
covering the LINAC's cryogenics (CMA01-12, CMB01-07 cryomodules and cryo plants),
vacuum, beam-position/loss monitors (BPM/BLM), RFQ RF chain, and injector/source
beamline. 4,496 PVs across 148 archiver groups.

## Files

- **`pv_inventory_raw.csv`** (4,497 rows) — every `(archive_port, group, pv_name)`
  triple, scraped directly from each archiver engine's web interface. Columns:
  `port`, `group`, `channel`.
- **`pv_inventory_categorized.csv`** (4,497 rows) — the same inventory with a
  `category` column added (RF/LLRF, Vacuum, BPM, BLM, Cryo variants, Pickup electron,
  etc.), assigned by group-name/PV-name pattern matching and cross-checked against
  GANIL control-system expert input. 16 PVs remain unresolved (`LME-FH21/22/23`).
  Columns: `archive_port`, `group`, `pv_name`, `category`.
- **`extraction_list_clean.csv`** (24,980 rows) — the categorized inventory expanded
  per cryomodule: which PVs were actually pulled for each of the 19 cryomodules'
  fault-precursor time windows (shared diagnostic channels like BPM/BLM/Vacuum are
  replicated per module since they're LINAC-wide, not attributable to one
  cryomodule). 2,620 distinct PVs. Columns: `module`, `category`, `archive_port`,
  `group`, `pv_name`.

## Known caveat

74 injector/source-line vacuum PVs (`LBE1-CF11/CF13`, `LBE1-SI1-TURBO`,
`LBE1-VIDE`, `LBE2-CF11/CF12/TURBO/VIDE`, `LBEC-VIDE`) leaked into the "Vacuum"
category via an over-broad classification pattern and are present in every
module's row set in `extraction_list_clean.csv`. Left as-is rather than
reprocessing.

## Context

This inventory fed a precursor-signal investigation correlating slow-control
telemetry with LLRF faults — see the
[Precursor Root-Cause Atlas](https://aghribi.github.io/LLRF_SP2_pipeline/precursor-dashboard.html)
and the [Heat-Load Observer dashboard](https://aghribi.github.io/LLRF_SP2_pipeline/heatload-observer-dashboard.html).
Key finding from the survey itself: no per-cavity RF amplitude/phase PV exists for
the CMA/CMB cryomodules in this archiver — only the RFQ has slow-archived `Llrf.*`
fields, plus 8 LME-line pickup channels. Everything else archived for CMA/CMB is
cryogenics or cavity-vacuum-barrier status, not the RF field itself.
