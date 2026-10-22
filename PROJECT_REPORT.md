# Hybrid Deep-Learning Network Intrusion Detection System (NIDS)

## Project Report

This document consolidates the design, implementation, methodology, experiments, tuning results, and report artifacts for the hybrid NIDS project developed in this repository. It is based on the source code and saved outputs present in the workspace, including the training pipeline, report files, tuning results, and dashboard implementation.

## 1. Executive Summary

The repository implements a hybrid intrusion detection framework that combines:

- Engine A: a benign-only flow autoencoder for anomaly detection,
- Engine B: clustering of anomalous embeddings to profile suspicious behavior,
- Engine C: day-based supervised specialists for known attacks,
- temporal generalization analysis and a dashboard for reporting.

The design follows a practical cybersecurity principle: known attacks should be handled by supervised specialists, while novel or drifted attacks should be surfaced by anomaly-based detection. This is especially important for CICIDS2017, where attack traffic changes across days and labels are not uniformly present across all temporal windows.

The project contains multiple layers of analysis:

1. raw preprocessing and auditing of CICIDS2017 CSV files,
2. feature engineering and leakage-safe splitting,
3. anomaly detection using a flow FT-Transformer autoencoder,
4. unsupervised anomaly clustering,
5. supervised specialist classifiers per traffic day,
6. hyperparameter tuning via isolated random search,
7. reporting artifacts and a Streamlit dashboard.

## 2. Problem Context

Traditional intrusion detection systems often rely on supervised classification of known attack families. This works well only when the deployed traffic distribution matches the historical training distribution. In real networks, however:

- attack behavior evolves over time,
- new families appear,
- traffic volume shifts,
- feature distributions drift,
- and benign traffic remains the most stable signal for normal operation.

The repository addresses this by separating the tasks:

- learn normal behavior from benign traffic,
- detect deviations as anomalies,
- cluster anomalies into similar behavioral groups,
- classify known attack patterns using day-specific specialists.

This hybrid strategy is well suited to modern NIDS deployments where full attack coverage cannot be guaranteed in advance.

## 3. Objectives

The project was designed to achieve the following objectives:

- Clean and validate CICIDS2017 generated network-flow data.
- Remove identifiers and leakage-prone metadata before training.
- Fit all preprocessing and feature-selection pipelines only on the train split.
- Build a benign-only anomaly detector for network flows.
- Calibrate anomaly thresholds from benign validation behavior.
- Cluster suspicious flows into anomaly families without using attack labels as training targets.
- Train day-specific specialist models for known attacks.
- Evaluate random and time-ordered generalization.
- Save reproducible tuning artifacts for hyperparameter search.
- Provide a dashboard and report outputs for interpretation.

## 4. Dataset Description

### 4.1 Source dataset

The project uses the CICIDS2017 dataset, a labelled traffic dataset generated for intrusion detection evaluation. It includes traffic for multiple working days, each with a different pattern of malicious and benign behavior.

The directory structure shows raw extracted flows in:

- `extracted_generated_flows/TrafficLabelling/`
- and processed artifacts in `production_hybrid_nids/cache/`

### 4.2 Daily traffic composition

The report files and code show the following traffic structure:

| Day | Main role | Labels observed |
|---|---|---|
| Monday | benign baseline | BENIGN |
| Tuesday | known attack traffic | BENIGN, FTP-Patator, SSH-Patator |
| Wednesday | DoS and infrastructure attack patterns | BENIGN, DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest, Heartbleed |
| Thursday | web/infiltration attack scenarios | BENIGN, Infiltration, Web Attack - Brute Force, Web Attack - XSS, Web Attack - Sql Injection |
| Friday | high-volume and scanning traffic | BENIGN, DDoS, PortScan, Bot |

### 4.3 Observed class counts

The saved dataset summary in `reports/label_counts_by_day.csv` reports the following counts:

| Label | Monday | Tuesday | Wednesday | Thursday | Friday |
|---|---:|---:|---:|---:|---:|
| BENIGN | 529918 | 432074 | 440031 | 456752 | 414322 |
| Bot | 0 | 0 | 0 | 0 | 1966 |
| DDoS | 0 | 0 | 0 | 0 | 128027 |
| DoS GoldenEye | 0 | 0 | 10293 | 0 | 0 |
| DoS Hulk | 0 | 0 | 231073 | 0 | 0 |
| DoS Slowhttptest | 0 | 0 | 5499 | 0 | 0 |
| DoS slowloris | 0 | 0 | 5796 | 0 | 0 |
| FTP-Patator | 0 | 7938 | 0 | 0 | 0 |
| Heartbleed | 0 | 0 | 11 | 0 | 0 |
| Infiltration | 0 | 0 | 0 | 36 | 0 |
| PortScan | 0 | 0 | 0 | 0 | 158930 |
| SSH-Patator | 0 | 5897 | 0 | 0 | 0 |
| Web Attack - Brute Force | 0 | 0 | 0 | 1507 | 0 |
| Web Attack - Sql Injection | 0 | 0 | 0 | 21 | 0 |
| Web Attack - XSS | 0 | 0 | 0 | 652 | 0 |

This confirms that Monday is dominated by benign traffic, while other days contain attack flows with different distributions.

### 4.4 Data integrity audit

The file audit file `reports/file_audit.csv` confirms that the extracted CSV files are consistent and that date parsing succeeded for all files. The summary indicates there are no broken date columns or missing timestamps. It also shows that Thursday has a file with a partial duplicate overlap, which the project specifically calls out as a historical issue that was fixed during the data cleaning stage.

## 5. Data Cleaning and Preprocessing Methodology

The repository contains a careful preprocessing pipeline designed to minimize leakage. The key principle is: train-time transformations must be fit only using the training split.

### 5.1 Leakage-safe design

The pipeline performs the following sequence:

1. Load raw CSVs.
2. Normalize column names and labels.
3. Remove metadata and leakage-prone fields.
4. Replace infinities with missing values.
5. Convert values to numeric types.
6. Replace missing values with medians computed on the training split.
7. Standardize continuous features using a scaler fit on the train split.
8. Apply feature ranking and selection using train-only data.
9. Save processed artifacts to parquet and NumPy arrays.

This approach is essential because flow metadata, timestamps, and identifying fields can leak label-related structure into the model if not removed.

### 5.2 Feature selection approach

The code uses a dual ranking method combining:

- Random Forest feature importance,
- Mutual Information (MI) scores.

The script combines the ranks of these measures and keeps the top-k features so that the selected feature set is targeted toward discriminative traffic behavior while still being compact.

### 5.3 Saved feature artifacts

The project stores the main selected feature set and related pipeline objects in:

- `production_hybrid_nids/features/selected_features.json`
- `production_hybrid_nids/features/feature_ranking.csv`
- `production_hybrid_nids/features/specialist_<Day>/pipeline.joblib`

The artifact `reports/k_selection.csv` reports experiments on candidate feature counts, with the following values:

| k | val_AP | val_ROC_AUC |
|---:|---:|---:|
| 10 | 0.0067 | 0.4858 |
| 20 | 0.0163 | 0.7791 |
| 26 | 0.0106 | 0.5549 |
| 30 | 0.2816 | 0.9801 |
| 40 | 0.3422 | 0.9792 |
| 44 | 0.2961 | 0.9752 |

This suggests the system identified a practical operating range in which a compact but effective feature subset is retained.

## 6. System Architecture Overview

The project contains a hybrid NIDS architecture organized into three major engines and a supporting evaluation layer.

### 6.1 Engine A: anomaly detection

Engine A learns only from benign traffic. It detects flows that differ from the benign distribution. This gives it an ability to surface zero-day or novel malicious behavior without requiring attack labels.

### 6.2 Engine B: anomaly profiling

Engine B receives suspicious flows from Engine A and clusters them in embedding space. This helps identify groups of anomalous traffic that share behavior, which is useful for profiling suspicious families, even when labels are unavailable.

### 6.3 Engine C: supervised day specialists

Engine C trains separate classifiers for each day with attacks. This is a realistic design because attack distribution varies by day and the traffic type is not perfectly homogeneous across the dataset.

### 6.4 Temporal generalization

A section of the project compares train/test framing under random split and chronological evaluation. This is essential because a time-ordered split is a stronger indicator of real deployment value than a random split, which can overestimate performance by mixing future and past events together.

## 7. Engine A: Benign-Only Flow Autoencoder

### 7.1 Architectural concept

The anomaly detector is a flow-level FT-Transformer autoencoder. The core idea is to reconstruct the transformed benign feature vector and flag large reconstruction errors.

The flow feature representation is fed into a tokenization layer and a transformer-style encoder. The latent bottleneck captures the most meaningful benign behavior, and the decoder reconstructs the original vector.

### 7.2 Model formulation

The project uses a lightweight hybrid architecture based on:

- feature-tokenization,
- multi-head self-attention,
- dense bottleneck,
- reconstruction objective,
- AdamW optimization.

It is designed to model feature interactions and nonlinear structure without requiring attack labels.

### 7.3 Threshold calibration

The system calibrates thresholds using benign validation reconstruction errors. This avoids making the model dependent on attack samples for threshold setting. Common threshold metrics include:

- mean reconstruction error,
- top-k error scores,
- max error values,
- percentile-based thresholds such as p90, p95, p97, p99, and p99.5.

The saved threshold artifact shows that the project performs threshold tuning with statistical calibration rather than hardcoded values.

## 8. Engine B: Unsupervised Anomaly Clustering

### 8.1 Purpose

Once anomalous flows are detected, Engine B groups them into clusters based on their learned embeddings. This helps human analysts understand whether suspicious traffic appears as one large family or several distinct patterns.

### 8.2 Algorithms used

The project applies:

- KMeans for fixed cluster partitioning,
- HDBSCAN for density-based cluster discovery,
- post-hoc cluster auditing using labels to interpret group composition.

This makes Engine B useful both for anomaly profiling and for attack family understanding when labels are not known in advance.

### 8.3 Practical role

Engine B does not “classify” a traffic family in the supervised sense; instead, it profiles anomalous behavior clusters. In other words, it helps answer:

- Are these abnormal flows all the same mode of behavior?
- Are there multiple suspicious groups?
- Which families could be mapped to attacks later?

## 9. Engine C: Day-Specific Supervised Specialists

### 9.1 Architectural concept

Engine C trains one specialist model per traffic day with attacks. The design is motivated by the fact that CICIDS2017 has different traffic distributions by day, and the model should not try to learn every attack regime in one monolithic classifier if the data is naturally segmented by time.

### 9.2 Model structure

The model combines:

- FT-Transformer encoder for flow-level feature interactions,
- temporal convolutional blocks (TCN) to model dependencies across time windows,
- dense projection layers,
- Softmax output for multiclass attack classification.

The code implements a TimeDistributed encoder, then a TCN over the window sequence, followed by temporal pooling and a classifier head.

### 9.3 Training methodology

The day specialist pipeline does the following:

- load the clean day parquet,
- stratified 70/30 split for the day’s traffic,
- fit medians, feature ranking, and scaler on the training split only,
- transform the full day using the frozen train-only pipeline,
- generate sliding windows with a configurable lookback,
- build class-weighted training objective for imbalance,
- train with early stopping,
- save metrics, confusion matrix, and ROC/PR curves.

### 9.4 Class imbalance handling

The project explicitly uses class-weight balancing and applies a cap to the max class weight to avoid over-dominating rare classes. This matters because traffic classes such as `Heartbleed`, `Infiltration`, and `Web Attack - Sql Injection` can be underrepresented even when the dataset is large.

### 9.5 Engine C evaluation outputs

The project saves per-day metrics in the reports directory and includes:

- confusion matrices,
- ROC and precision-recall curves,
- classification report dictionaries,
- summary scorecards,
- per-day specialist metrics.

The `reports/model_comparison.csv` file contains model comparison values for candidate architectures. The file shows:

| Model | val_AP | val_AUC | test_AP | test_AUC | test_precision | test_recall | test_F1 | test_FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FT-Transformer | 0.0322 | 0.9296 | 0.8064 | 0.8895 | 0.7711 | 0.3791 | 0.5083 | 0.0785 |
| TCN | 0.1140 | 0.9564 | 0.8092 | 0.8514 | 0.9751 | 0.3316 | 0.4949 | 0.0059 |

These values illustrate a model comparison on a selected anomaly or classification task. They are useful as comparative evidence of the relative performance of the strong baselines explored during the project.

## 10. Hyperparameter Tuning Methodology

Hyperparameter tuning is implemented in `tune_hyperparameters.py` and supports isolated random-search for Engine A and Engine C.

### 10.1 Tuning philosophy

The script is deliberately isolated from the production pipeline. It does not overwrite the main models, reports, or feature pipelines. Instead, it writes results to the `tuning_results/` directory.

This is a good practice because it keeps the experiments reproducible and keeps the production implementation separate from the tuning experiments.

### 10.2 Tuning configuration

The tuner samples from a discrete set of values for parameters like:

- `lookback`,
- `batch_size`,
- `epochs`,
- `patience`,
- `d_token`,
- `n_heads`,
- `n_ft_blocks`,
- `tcn_filters`,
- `tcn_kernel_size`,
- `dropout`,
- `learning_rate`,
- `weight_decay`,
- `max_class_weight`.

### 10.3 Saved Tuesday best configuration

The current best saved configuration for Tuesday is in:

- `tuning_results/engine_c_tuesday_best_params.json`

The file shows a tuning trial with the following values:

```json
{
  "trial": 1,
  "day": "Tuesday",
  "status": "ok",
  "lookback": 10,
  "batch_size": 256,
  "epochs": 4,
  "patience": 3,
  "top_k_features": 32,
  "d_token": 32,
  "n_heads": 8,
  "n_ft_blocks": 2,
  "tcn_filters": 64,
  "tcn_kernel_size": 3,
  "tcn_dilations": [1, 2, 4, 8],
  "dropout": 0.1,
  "learning_rate": 0.001,
  "weight_decay": 1e-05,
  "max_class_weight": 250.0,
  "seed": 42,
  "smoke_test": true,
  "smoke_samples": 5000,
  "val_macro_f1": 1.0,
  "val_loss": 0.0,
  "objective": 1.0,
  "seconds": 83.65
}
```

This is an important result to interpret carefully:

- `smoke_test` is set to `true`,
- `epochs` = 4,
- `smoke_samples` = 5000,
- full objective metrics = 1.0.

This indicates that the saved Tuesday result is a quick smoke-test run, not a final production-calibrated training run. It is still useful as an experiment artifact, but it should not be treated as a final deployment metric without a full non-smoke retraining pass.

## 11. CLI and Execution Commands

The repository exposes a canonical project launcher in `run_hybrid_nids.py`.

### 11.1 Full pipeline

```powershell
& "D:\DEEP LEARNING PROJECT\.venv\Scripts\python.exe" `
  "D:\DEEP LEARNING PROJECT\DEEP LEARNING PROJECT\run_hybrid_nids.py" `
  --mode all
```

This runs the full pipeline: Engine A -> Engine B -> Engine C -> temporal evaluation.

### 11.2 Engine A only

```powershell
& "D:\DEEP LEARNING PROJECT\.venv\Scripts\python.exe" `
  "D:\DEEP LEARNING PROJECT\DEEP LEARNING PROJECT\run_hybrid_nids.py" `
  --mode engine_a
```

### 11.3 Engine B only

```powershell
& "D:\DEEP LEARNING PROJECT\.venv\Scripts\python.exe" `
  "D:\DEEP LEARNING PROJECT\DEEP LEARNING PROJECT\run_hybrid_nids.py" `
  --mode engine_b
```

### 11.4 Engine C all days

```powershell
& "D:\DEEP LEARNING PROJECT\.venv\Scripts\python.exe" `
  "D:\DEEP LEARNING PROJECT\DEEP LEARNING PROJECT\run_hybrid_nids.py" `
  --mode engine_c
```

### 11.5 Engine C for Tuesday only

```powershell
& "D:\DEEP LEARNING PROJECT\.venv\Scripts\python.exe" `
  "D:\DEEP LEARNING PROJECT\DEEP LEARNING PROJECT\run_hybrid_nids.py" `
  --mode engine_c --day Tuesday
```

### 11.6 Hyperparameter tuning for Tuesday only

```powershell
& "D:\DEEP LEARNING PROJECT\.venv\Scripts\python.exe" `
  "D:\DEEP LEARNING PROJECT\DEEP LEARNING PROJECT\tune_hyperparameters.py" `
  --engine c --day Tuesday --trials 10
```

### 11.7 Smoke-test tuning run

```powershell
& "D:\DEEP LEARNING PROJECT\.venv\Scripts\python.exe" `
  "D:\DEEP LEARNING PROJECT\DEEP LEARNING PROJECT\tune_hyperparameters.py" `
  --engine c --day Tuesday --trials 10 --smoke-test
```

## 12. Dashboard and Reporting

The repository includes a Streamlit dashboard in `dashboard/app.py`.

It provides:

- overview of project status,
- CSV audit and prediction access,
- Engine A and B explorations,
- Engine C summaries,
- tuning trial exploration,
- and best-configuration display.

This makes it easy to inspect results without opening raw notebooks or scripts.

## 13. Project Structure

The repository is organized as follows:

```text
.
├── README.md
├── PROJECT_REPORT.md
├── run_hybrid_nids.py
├── tune_hyperparameters.py
├── dashboard/
│   ├── app.py
│   ├── README.md
│   └── requirements.txt
├── production_hybrid_nids/
│   ├── engine_a_flow_autoencoder.py
│   ├── engine_b_clustering.py
│   ├── engine_c_specialists.py
│   ├── section5_temporal_generalization.py
│   ├── cache/
│   ├── clusters/
│   ├── features/
│   ├── models/
│   └── reports/
├── extracted_generated_flows/
├── processed/
├── raw/
├── reports/
├── tuning_results/
├── artifacts/
├── models/
└── Colab Notebooks/
```

## 14. Key Implementation Observations

### 14.1 Strengths

- clean separation of engines,
- modular training and evaluation pipeline,
- leakage-safe feature selection,
- strong experimentation discipline,
- quality report generation,
- reproducible tuning outputs,
- ability to run the full hybrid system via a single command.

### 14.2 Gaps and caveats

The project is strong, but several important caveats remain:

1. The saved Tuesday best-params file is a smoke-test result rather than a full production fitness run.
2. The tuning output is not automatically fed back into the production model route.
3. A fully finalized deployment pipeline would require one more step: apply best config, retrain, and validate without smoke mode.
4. The report files are more complete than the final production run summary in some cases, so there is a distinction between experimentation artifacts and official final metrics.

This does not invalidate the approach; it simply indicates that the project is at an advanced experimental stage with a clean framework but a final productionization pass is still needed.

## 15. Final Assessment

This project successfully demonstrates a thoughtful hybrid approach to NIDS with strong engineering design:

- benign-only anomaly detection for unfamiliar attack behavior,
- clustering for anomaly profiling,
- supervised specialists for known classes,
- leakage-aware preprocessing,
- reproducible random search for hyperparameters,
- and a dashboard for inspection.

The project is a solid research and engineering prototype and is already organized in a way that can be extended toward a real production deployment pipeline. The next logical step would be to formalize the final retraining loop based on the selected best hyperparameters and to generate a single official production summary report.

## 16. Conclusion

The repository implements a complete end-to-end hybrid NIDS workflow grounded in real CICIDS2017 traffic and a structured codebase. It covers data preparation, anomaly detection, clustering, day-specific classification, feature selection, temporal evaluation, tuning, and reporting.

It is best understood as a robust experimental and educational system that is close to deployment-ready, but not yet fully finalized as a single production stack that automatically consumes the best tuning configuration for final retraining.

---

This report was derived from the project files, saved artifacts, and generated outputs present in the workspace, including the source implementations in `run_hybrid_nids.py`, `tune_hyperparameters.py`, `production_hybrid_nids/engine_c_specialists.py`, `dashboard/app.py`, and the generated files in `reports/` and `tuning_results/`.
