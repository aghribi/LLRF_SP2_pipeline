# Pre-treatment/pre-selection comparison: Lassalle thesis vs. current pipeline

Date: 2026-09-03. Source: `ipac2026/poster_ipac2026.tex`, `ipac2026/outline_ipac2026.tex` (both cite
thesis sections/figures directly), and `pipeline/00_scripts/prepare_data_cluster.py`. Thesis PDFs
themselves (`inputs/CharlyLassalle_these_anomalies_18122025.pdf`,
`../ipac2026/ManuscritPhD_CharlyLASSALLE2.pdf`) could not be read directly in this environment (no
PDF text-extraction tooling available) — everything below is sourced from the poster/outline's
explicit citations of thesis content, not from the PDF itself. Read the PDF directly to verify
before acting on the most consequential item (label reliability).

## Lassalle's pre-treatment (per `outline_ipac2026.tex` §3.1–3.2)

- Quality filters: `LOOP = ON`, `BEAM = OUI`, `KPI ≥ 10`, mean(Ucav)|pre-trigger ≥ 0.1 MV/m. NDEC fixed at 200 (100,096-sample files, 88.05 kHz).
- **Label reliability check**: cross-checked the header `ALM` field against the per-sample "défauts" register and found **~38% disagreement** between the two. Labels were derived from the waveform-level fault register instead of the header field — called out explicitly as "a key finding" of the thesis.
- Preprocessing: temporal windowing, optional 4x downsampling, per-file per-signal local normalization (z-score or min-max), correlation-based feature pruning, tsfresh extraction with low-variance removal + Pearson de-dup + SelectKBest.

## Current anomalies_exploration pre-treatment (per `prepare_data_cluster.py`)

- `passes_quality_filters()` (lines ~193–286): `KPI ≥ 10`, mean(|Ucav|) ≥ 0.1, mean(|Uci|) ≥ 0.02 (the latter two added 2026-01-17, commit `eec73be`, "RF power filters"). **No LOOP or BEAM check.** NDEC filter was explicitly removed (comment at ~lines 272–284: "CRITICAL FIX: Removed NDEC filter... Now accepts ALL NDEC values").
- `extract_fault_labels()` (lines ~289–311): derives labels **solely from the header ALM field**. `Read_Signals(..., compute_defauts=False, compute_etats=False, ...)` — the per-sample "défauts" register that PyPostMortem can compute is never requested.
- `preprocess_signals()` (lines ~314–376): DC-removal high-pass Butterworth + global z-score, with a **hardcoded `trigger_idx = 3000` / `delta_pre = 3000`** regardless of each file's actual `POSTROW`.
- `GetCaviteCoef()` (per-cavity calibration, defined in `programmes/PyPostMortem`) is **never called** — 0 hits across all copies of the extraction script — despite being documented as intended in `docs/ANOMALY_DETECTION_STRATEGY.md` §2.1.
- No deduplication, no outlier clipping, no POSTROW-based dynamic alignment.

## Gap list (highest impact first)

1. **Label reliability never cross-checked.** This is the important one: if the ~38% ALM/défauts disagreement Lassalle found on his dataset also applies here, the ALM-only labels currently used as ground truth for every supervised model in this pipeline (binary, multi-label, root-cause) carry substantial label noise. This isn't a display bug — it would understate/distort every reported accuracy/F1/AUC in `report/`. **Recommended action: verify by computing `Read_Signals(..., compute_defauts=True)` on a sample of events and measuring ALM-vs-défauts agreement on this dataset before trusting current model metrics.**
2. **No LOOP/BEAM gating** — current filter can admit events where the RF loop wasn't actually closed or beam wasn't present, which the Jan-17 RF-power filters partially but not fully address (Ucav/Uci thresholds are a proxy, not a direct LOOP/BEAM check).
3. **No per-cavity calibration** (`GetCaviteCoef` unused) — raw ADC-scale signals may not be physically comparable across cavities, contrary to the pipeline's own documented intent.
4. **Fixed NDEC removed but alignment still assumes fixed sample offsets** — `trigger_idx=3000` may misalign events acquired at different decimation factors now that NDEC is no longer restricted to 200.
5. **No POSTROW-based dynamic alignment or deduplication**, both present conceptually in the shared `PyPostMortem` library but unused here.

## Not verified

- Exact thesis chapter/page content (PDF unreadable in this environment — retry with a PDF-capable tool or read manually).
- Whether `utilities/data_preparation/` or `archive/maintenance_scripts/` contain alternate versions that already address any of these gaps (only greped, not fully read).
