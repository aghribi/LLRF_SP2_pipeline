**Pipeline analysis & adapted diagram (generalized)**

- **Summary:** The pipeline is organized as a staged workflow that transforms raw data into model artifacts and reports. Key logical steps are: data preprocessing → feature engineering → anomaly detection (multiple methods run in parallel) → decision branching → optional classification and explainability → artifacts & monitoring. Precursor analysis can run independently from the features stage.

- **Files of interest:**
  - `pipeline/diagrams/pipeline_diagram.mmd` — Mermaid flowchart (generalized view).
  - `pipeline/00_scripts/` — preparation scripts (`prepare_*.py`, `prepare_data_cluster.py`).
  - `pipeline/01_slurm/` — SLURM submit scripts (`submit_*.sh`).
  - `pipeline/02_config/requirements_cluster.txt` — cluster dependency hints.

- **Generalized flow (conceptual):**
  1. Data Preprocessing — e.g. normalization, filtering, segmentation, quality filters, batching, checkpointing.
  2. Feature Engineering & Extraction — statistical features (mean, std, skewness, RMS, peaks), domain/physics features (Q_L, detuning), temporal segmentation, derivative-based features, PCA/embedding.
  3. Anomaly Detection — run several detection methods in parallel (e.g. Autoencoders, IsolationForest, sequence models like LSTM/CNN, ensemble/hybrid approaches).
  4. Decision Branch — if anomaly detected: proceed to classification and explainability; if not: store artifacts and logs.
  5. Anomalies Classification (optional) — simple binary label, multilabel decomposition, training pipelines and architectures for supervised models.
  6. Interpretability / Explainability — SHAP/LIME, saliency maps, feature importance reporting.
  7. Precursor Analysis (optional) — early-warning signals and trend detection, can be computed directly from the features or run as a separate analysis.
  8. Artifacts & Monitoring — models, serialized pipelines, reports, metrics, and job logs.

  
**Detailed methods (detection & classification)**

- **Anomaly detection methods (examples):**
  - Classical / Statistical:
    - Statistical thresholds (z-score, moving-window statistics, control charts)
    - Isolation Forest, One-Class SVM, Local Outlier Factor
  - Advanced / Model-based:
    - PCA / Projection and reconstruction-error methods
    - Shallow autoencoders and reconstruction-based detectors
    - Ensembles & hybrid detectors combining statistical and model outputs
  - Deep learning / Sequence models:
    - LSTM / GRU-based sequence anomaly detectors
    - Temporal CNNs / TCNs for time-series patterns
    - Transformer / Seq2Seq / attention-based architectures for long-range dependencies

- **Anomaly classification methods (examples):**
  - Classical:
    - Logistic Regression, Support Vector Machines, Random Forests
  - Advanced / Gradient-boosted:
    - XGBoost, LightGBM, CatBoost (effective on tabular engineered features)
  - Deep learning:
    - MLPs on feature vectors, CNN/RNN architectures for raw/temporal inputs, Transformer-based classifiers

These choices can be mixed: e.g., use an Autoencoder (deep) for detection, then use gradient-boosted trees for classification on engineered features; or apply ensemble voting across multiple detectors.

- **Notes & practical pointers:**
  - `prepare_data_cluster.py` in `pipeline/00_scripts` uses per-file timeouts, multiprocessing, and save/merge batches to avoid OOM.
  - Feature outputs are saved as `features_engineered.pkl` (contains `X_scaled`, `X_pca`, labels, scaler/pca objects).
  - SLURM scripts assume a conda env named `anomalies` and cluster dataset paths; adjust `DATA_DIR`/`OUTPUT_DIR` for local runs.

- **How to render the Mermaid diagram to PNG (example):**
  - Using Mermaid CLI (preferred):

```bash
cd anomalies_exploration
npx -y @mermaid-js/mermaid-cli -i pipeline/diagrams/pipeline_diagram.mmd -o pipeline/diagrams/pipeline_diagram.png
```

  - Or with `mmdc` if installed system-wide:

```bash
mmdc -i pipeline/diagrams/pipeline_diagram.mmd -o pipeline/diagrams/pipeline_diagram.png
```

- **Next steps (optional):**
  - Render the diagram to PNG/SVG in-repo (I will attempt this now).
  - Add a short `pipeline/README.md` with run examples and SLURM tuning tips.
  - Generate a Makefile or lightweight orchestrator to run stages in sequence.

Files:
- `pipeline/diagrams/pipeline_diagram.mmd`
- `pipeline/PIPELINE_DIAGRAM.md` (this file)
