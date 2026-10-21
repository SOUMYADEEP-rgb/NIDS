# Hybrid Deep-Learning Network Intrusion Detection System

## Project Report

**Dataset:** CICIDS2017

**Project type:** Hybrid deep-learning and unsupervised network intrusion detection system

**Main components:** Engine A anomaly detection, Engine B anomaly clustering, Engine C supervised attack classification, and temporal generalization analysis

**Implementation:** Python, TensorFlow/Keras, scikit-learn, Pandas, NumPy, Parquet, Streamlit

**Report status:** Based on the current source code, saved artifacts, generated reports, tuning extension, and dashboard in this repository.

---

## 1. Abstract

Traditional network intrusion detection systems are often trained as supervised classifiers. They perform well when the attack categories seen during deployment are already present in the training data, but they can fail when a new attack appears. The objective of this project is to develop a hybrid Network Intrusion Detection System (NIDS) that combines supervised classification with zero-day anomaly detection and unsupervised attack profiling.

The system uses CICIDS2017 network-flow data. Monday benign traffic is used to learn normal behavior in Engine A. Engine A is a flow-level FT-Transformer autoencoder that identifies unusual flows through reconstruction error. Suspicious flows are passed to Engine B, which uses KMeans and HDBSCAN on learned embeddings to group previously unseen traffic by behavior. Engine C consists of day-specific FT-Transformer and Temporal Convolutional Network (TCN) classifiers for known attack categories. A temporal-generalization module compares random stratified evaluation with chronological deployment-like evaluation to study concept drift.

The project demonstrates that a hybrid architecture is more suitable for changing network conditions than a supervised classifier alone. Engine C provides detailed labels for known attacks, while Engine A and Engine B provide a detection and profiling path for traffic that is not represented in supervised training.

---

## 2. Problem Statement

Network traffic changes over time. New attacks, altered attack behavior, and changes in traffic volume can make a classifier trained on historical labels unreliable. The project addresses the following questions:

1. Can normal network behavior be learned from benign traffic only?
2. Can anomalous traffic from unseen days be detected without training on its attack labels?
3. Can suspicious flows be grouped into meaningful behavioral clusters?
4. Can known attacks still be classified accurately with supervised specialist models?
5. How much does performance change when the test traffic occurs later in time than the training traffic?

---

## 3. Objectives

- Build a leakage-safe preprocessing pipeline for CICIDS2017.
- Remove identifiers that could cause memorization or leakage.
- Select useful behavioral features using Random Forest and Mutual Information.
- Train a benign-only autoencoder for anomaly detection.
- Calibrate anomaly thresholds using a benign validation split.
- Cluster anomaly embeddings without using attack labels for model fitting.
- Train day-specific supervised specialists for known attacks.
- Evaluate both random and chronological generalization.
- Add reproducible hyperparameter tuning for Engine A and Engine C.
- Provide a dashboard for reporting and demonstration.

---

## 4. Dataset

The project uses the CICIDS2017 generated labelled-flow dataset. The source data contains approximately 2.83 million network flows collected over five working days.

### 4.1 Day-wise attack distribution

| Day | Traffic role | Main labels |
|---|---|---|
| Monday | Benign baseline | BENIGN |
| Tuesday | Known attacks | BENIGN, FTP-Patator, SSH-Patator |
| Wednesday | Known DoS attacks | BENIGN, DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest, Heartbleed |
| Thursday | Unseen web/infiltration traffic | BENIGN, Infiltration, Web Attack - Brute Force, Web Attack - XSS, Web Attack - Sql Injection |
| Friday | Unseen high-volume and scanning traffic | BENIGN, DDoS, PortScan, Bot |

Monday is especially important because it supplies the benign-only training distribution for Engine A. Thursday and Friday contain attack behavior that is not used as labelled training data for Engine A.

### 4.2 Raw data locations

- Raw archive: `raw/GeneratedLabelledFlows.zip`
- Extracted CSV files: `extracted_generated_flows/TrafficLabelling/`
- Cleaned Parquet caches: `production_hybrid_nids/cache/`
- NumPy model arrays: `production_hybrid_nids/cache/X_*.npy`

---

## 5. Data preprocessing methodology

The preprocessing pipeline performs the following operations:

1. Read the CICIDS2017 CSV files.
2. Normalize column names and clean labels.
3. Remove socket and metadata columns, including source/destination identifiers, ports, timestamps, protocol, flow ID, and label from model features.
4. Convert model columns to numeric values.
5. Replace positive and negative infinity with missing values.
6. Handle missing values with medians calculated from the relevant training split.
7. Parse and normalize timestamps.
8. Sort flows chronologically.
9. Split data before fitting feature selectors, medians, scalers, or models.
10. Cache the processed results as Parquet and NumPy files.

This ordering is important. If scaling or feature selection were fitted on the complete dataset before splitting, information from validation or test traffic could leak into training.

### 5.1 Feature selection

Feature selection combines:

- Random Forest feature importance.
- Mutual Information with the binary attack/benign target.

The two rankings are combined and the highest-ranked features are retained. The main production pipeline uses 32 selected features. Engine C also stores day-specific feature-selection artifacts.

Important artifacts:

- `production_hybrid_nids/features/selected_features.json`
- `production_hybrid_nids/features/feature_ranking.csv`
- `production_hybrid_nids/features/scaler.joblib`
- `production_hybrid_nids/features/train_medians.joblib`
- `production_hybrid_nids/features/specialist_<Day>/pipeline.joblib`

---

## 6. Overall system architecture

```text
Raw CICIDS2017 network flows
             |
             v
Cleaning, validation, scaling, feature selection
             |
             v
      32 behavioral features
             |
     +-------+--------+
     |                |
     v                v
 Engine A          Engine C
 Anomaly           Day-specific
 detection        known-attack
     |             classification
     v                |
 Suspicious          v
 flows          Known attack label
     |
     v
 Engine B
 KMeans + HDBSCAN
 anomaly profiling
```

The engines have different responsibilities:

- Engine A answers: **Does this flow look unlike normal traffic?**
- Engine B answers: **Which suspicious flows behave similarly?**
- Engine C answers: **Which known attack class does this flow belong to?**

---

## 7. Engine A: benign-only anomaly detection

Source implementation: `production_hybrid_nids/engine_a_flow_autoencoder.py`

### 7.1 Purpose

Engine A is trained on Monday BENIGN flows only. It learns to reconstruct normal flow features. When an input is unlike the learned benign distribution, its reconstruction error is expected to increase.

This design allows Engine A to detect traffic from attack families that were not supplied as supervised training labels.

### 7.2 Architecture

```text
32 continuous input features
          |
          v
Feature tokenizer
          |
          v
FT-Transformer blocks
          |
          v
Global average pooling
          |
          v
64-dimensional bottleneck
          |
          v
Dense decoder
          |
          v
Reconstructed 32 features
```

The tokenizer maps every scalar feature into a learned token vector. Multi-head self-attention models relationships between features. The bottleneck is a compact representation of the flow. The decoder attempts to reconstruct the original transformed feature vector.

### 7.3 Engine A hyperparameters

| Hyperparameter | Production value |
|---|---:|
| Number of features | 32 |
| Token dimension | 48 |
| Attention heads | 4 |
| Transformer blocks | 2 |
| Bottleneck dimension | 64 |
| Dropout | 0.05 |
| Learning rate | 0.0007 |
| Weight decay | 0.00001 |
| Batch size | 1024 |
| Maximum epochs | 12 |
| Early-stopping patience | 4 |
| Optimizer | AdamW |
| Loss | Mean Squared Error |
| Seed | 42 |

### 7.4 Threshold calibration

The Monday data is divided into an 80/20 training/validation split. Thresholds are calculated only from the benign validation reconstruction errors.

The saved threshold artifact is:

`production_hybrid_nids/models/engine_a_flow_ae/thresholds_monday_val.json`

The available percentiles are p90, p95, p97, p99, and p99.5 for:

- Mean feature MSE.
- Top-3 feature MSE.
- Maximum feature MSE.

The production anomaly decision uses mean and top-3 error signals. A flow is flagged when the selected error exceeds its benign calibration threshold.

### 7.5 Engine A saved models

Located in `production_hybrid_nids/models/engine_a_flow_ae/`:

- `best_flow_ft_autoencoder.keras`
- `flow_ft_autoencoder.keras`
- `training_history.csv`
- `thresholds_monday_val.json`

---

## 8. Engine B: unsupervised anomaly profiling

Source implementation: `production_hybrid_nids/engine_b_clustering.py`

### 8.1 Purpose

Engine B receives the anomaly flows detected by Engine A. It extracts the 64-dimensional bottleneck representation and clusters suspicious flows according to learned behavioral similarity.

Attack labels are not required to fit KMeans or HDBSCAN. Labels are used only afterward for audit and cluster-purity analysis.

### 8.2 Algorithms

#### KMeans

KMeans partitions the anomaly embeddings into a fixed number of clusters. The current production value is `k = 5`.

#### HDBSCAN

HDBSCAN finds density-based groups and marks isolated points as noise. It does not require a fixed number of clusters.

### 8.3 Engine B hyperparameters

| Hyperparameter | Value |
|---|---:|
| KMeans clusters | 5 |
| KMeans `n_init` | 10 |
| HDBSCAN minimum cluster size | 50 |
| HDBSCAN minimum samples | 15 |
| Maximum clustering samples | 25,000 |
| Random state | 42 |

### 8.4 Engine B saved outputs

Located in `production_hybrid_nids/clusters/`:

- `kmeans_model.joblib`
- `kmeans_unseen_attacks.joblib`
- `clustered_anomalies.parquet`
- `cluster_evaluation.json`
- `cluster_feature_profiles.csv`
- `cluster_manifold_projection.png`

---

## 9. Engine C: supervised day-specific classification

Source implementation: `production_hybrid_nids/engine_c_specialists.py`

### 9.1 Purpose

Engine C performs detailed classification for attack classes that are represented in the training data. Separate specialists are trained for Tuesday, Wednesday, Thursday, and Friday.

This avoids forcing one classifier to learn all day-specific distributions at once.

### 9.2 Architecture

```text
Selected flow features
          |
          v
Sliding temporal windows, lookback = 10
          |
          +--------------------+
          |                    |
          v                    v
   FT-Transformer       Temporal CNN/TCN
   feature encoder      temporal encoder
          |                    |
          +---------+----------+
                    v
             Dense classification head
                    |
                    v
          Known attack probabilities
```

The FT-Transformer models feature interactions. The TCN models patterns across nearby flow windows. The final classification head produces multiclass probabilities.

### 9.3 Engine C hyperparameters

| Hyperparameter | Production value |
|---|---:|
| Lookback | 10 flows |
| Batch size | 512 |
| Maximum epochs | 20 |
| Early-stopping patience | 5 |
| Selected features | 32 |
| Token dimension | 48 |
| Attention heads | 4 |
| FT blocks | 2 |
| TCN filters | 96 |
| TCN kernel size | 3 |
| TCN dilations | 1, 2, 4, 8 |
| Dropout | 0.1 |
| Learning rate | 0.0005 |
| Weight decay | 0.0001 |
| Maximum class weight | 250 |
| Optimizer | AdamW |
| Loss | Sparse categorical cross-entropy |
| Split | 70/30 stratified |
| Seed | 42 |

Class weights are calculated from the training windows and capped to prevent very rare classes from producing unstable gradients.

---

## 10. Temporal generalization and concept drift

Source implementation: `production_hybrid_nids/section5_temporal_generalization.py`

Random stratified splits can make a model look strong because training and test samples come from similar time periods. A chronological split is closer to deployment: earlier flows are used for training and later flows are used for testing.

The project compares:

- Stratified random F1.
- Chronological F1.
- The difference between the two values.
- Hourly traffic and threat behavior.

This analysis is important because an attack may start after the model has already been trained for the day.

---

## 11. Experimental results

### 11.1 Engine A anomaly detection

The saved Engine A evaluation reports the following representative values:

| Evaluation day | Best score signal | ROC-AUC | Average Precision |
|---|---|---:|---:|
| Thursday | Top-3 feature MSE | 0.9005 | 0.0446 |
| Friday | Mean feature MSE | 0.8992 | 0.7194 |

The evaluation uses sampled flows from unseen days. The results show that the anomaly detector separates attack and benign behavior better than random ranking, especially for Friday traffic with a higher attack prior.

The report also shows 100% recall for the small Infiltration sample at the p99 threshold, but rare-class percentages should be interpreted carefully because the number of Infiltration examples is very small.

### 11.2 Engine B clustering

Current saved values in `production_hybrid_nids/clusters/cluster_evaluation.json`:

| Metric | Value |
|---|---:|
| KMeans clusters | 5 |
| Total clustered flows | 21,158 |
| Silhouette score | 0.5939 |
| Calinski–Harabasz index | 12,578.5 |
| Davies–Bouldin index | 0.9908 |
| HDBSCAN clusters found | 74 |
| HDBSCAN noise ratio | 19.30% |

KMeans cluster audit:

| Cluster | Size | Dominant behavior | Purity |
|---:|---:|---|---:|
| 0 | 7,651 | PortScan | 85.73% |
| 1 | 4,166 | DDoS | 98.44% |
| 2 | 1,945 | Web Attack - Brute Force/XSS family | 61.85% |
| 3 | 2,875 | DDoS burst variant | 91.69% |
| 4 | 4,521 | Benign/residual boundary | 62.20% |

The clustering results suggest that the learned anomaly embedding captures meaningful attack behavior. The weaker purity of Cluster 2 is expected because different web attack categories can share similar flow-level behavior, and Cluster 4 contains boundary traffic that is difficult to separate using a reconstruction threshold.

### 11.3 Engine C classification

The saved specialist summary currently contains:

| Day | Classes | Train windows | Test windows | Weighted F1 | Weighted precision | Weighted recall | Binary AP | Binary ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Tuesday | 3 | 312,127 | 133,764 | 0.9850 | 0.9887 | 0.9834 | 0.9406 | 0.9976 |
| Wednesday | 6 | 484,883 | 207,802 | 0.9812 | 0.9856 | 0.9793 | 0.9973 | 0.9988 |
| Thursday | 5 | 5,000 | 5,000 | 0.3432 | 0.9941 | 0.2106 | 0.0122 | 0.6240 |
| Friday | 4 | 5,000 | 5,000 | 0.9634 | 0.9676 | 0.9602 | 0.9775 | 0.9835 |

**Important reproducibility note:** The current saved Thursday and Friday specialist summary shows 5,000 training and 5,000 test windows, which indicates that those saved artifacts were generated using a smoke/subsampled run. Before using those two rows as final paper results, rerun the corresponding specialists with smoke testing disabled and regenerate the reports.

### 11.4 Temporal results

Current chronological comparison:

| Day | Stratified F1 | Chronological F1 | F1 difference |
|---|---:|---:|---:|
| Tuesday | 0.9850 | 0.9670 | 0.0180 |
| Wednesday | 0.9812 | 0.0000 | 0.9812 |
| Thursday | 0.3432 | 0.9999 | -0.6566 |
| Friday | 0.9634 | 0.0000 | 0.9634 |

These results demonstrate that random splits and chronological splits can measure very different properties. In particular, Wednesday and Friday show severe temporal regime changes. The chronological scores should be interpreted together with class balance and the hourly attack distribution, because a near-zero F1 can occur when the later window is dominated by a different traffic regime.

---

## 12. Main findings

### Finding 1: A benign-only model can provide a zero-day detection layer

Engine A does not require attack labels for training. Its reconstruction error identifies traffic that differs from the Monday benign baseline.

### Finding 2: Top-feature reconstruction error is useful

Mean error can dilute a strong deviation in one or two features. The top-3 feature error preserves the signal of severe deviations in a small number of flow features.

### Finding 3: Anomaly embeddings contain attack structure

Engine B produces meaningful clusters, including high-purity DDoS and PortScan groups. This supports the use of the autoencoder bottleneck as a behavioral representation.

### Finding 4: Supervised classification is strong for known distributions

Tuesday and Wednesday specialist models achieve approximately 98% weighted F1 in the saved full-size evaluations. Friday also has a high saved weighted F1, although the current summary uses a smoke-sized artifact and should be regenerated for final publication claims.

### Finding 5: Random evaluation can hide temporal failure

Chronological results demonstrate that traffic regimes can change during a day. A deployment-oriented NIDS therefore needs both supervised classification and an anomaly-detection fallback.

### Finding 6: The hybrid design has complementary responsibilities

Engine C explains known attacks. Engine A catches traffic outside the learned benign distribution. Engine B provides a first behavioral grouping for suspicious traffic that does not have a known label.

---

## 13. Hyperparameter-tuning extension

The project contains a separate tuning script:

`tune_hyperparameters.py`

It performs random search without modifying production models.

### 13.1 Engine A search space

- Token dimension: 32, 48, 64
- Attention heads: 2, 4, 8
- Transformer blocks: 1, 2, 3
- Bottleneck dimension: 32, 64, 96
- Dropout: 0.05, 0.1, 0.2
- Learning rate: 0.0003, 0.0007, 0.001
- Batch size: 512, 1024

The objective is the lowest Monday validation reconstruction loss.

### 13.2 Engine C search space

- Lookback: 1, 5, 10
- Batch size: 256, 512, 1024
- Token dimension: 32, 48, 64
- Attention heads: 2, 4, 8
- FT blocks: 1, 2, 3
- TCN filters: 64, 96, 128
- TCN kernel size: 3, 5, 7
- Dropout: 0.05, 0.1, 0.2, 0.3
- Learning rate: 0.0002, 0.0005, 0.001
- Weight decay: 0.00001, 0.0001, 0.001

The objective is the highest validation macro-F1.

### 13.3 Tuning results

Tuning outputs are stored in `tuning_results/`:

- `engine_a_trials.csv`
- `engine_a_best_params.json`
- `engine_c_<day>_trials.csv`
- `engine_c_<day>_best_params.json`

The current initial files were generated by smoke trials and should not be described as final optimized configurations. A full study should use multiple trials, ideally on a GPU, and then retrain the selected configuration before final test evaluation.

### 13.4 Correct model-selection protocol

```text
Training data
      |
      v
Trial-specific training
      |
      v
Validation metric
      |
      v
Select best hyperparameters
      |
      v
Retrain selected model
      |
      v
Evaluate once on untouched test data
```

The test set should not be used to select hyperparameters.

---

## 14. Dashboard

The dashboard is contained in `dashboard/` and is implemented with Streamlit.

Files:

- `dashboard/app.py`
- `dashboard/requirements.txt`
- `dashboard/README.md`

Dashboard pages:

1. Overview.
2. CSV prediction using Engine A.
3. Engine A anomaly visualization.
4. Engine B cluster evaluation and projection.
5. Engine C day-wise metrics and plots.
6. Hyperparameter tuning tables and best configurations.

The dashboard reads existing artifacts and reports. It does not retrain models when the page is refreshed.

---

## 15. Project dependencies

The configured virtual environment uses Python 3.13.5.

Main packages:

- TensorFlow 2.21.0
- Keras 3.15.1
- NumPy 2.5.4
- Pandas 3.0.6
- scikit-learn 1.9.1
- SciPy 1.18.1
- PyArrow 26.0.0
- Joblib 1.6.0
- Matplotlib 3.11.2
- Seaborn 0.13.2
- Streamlit for the dashboard

TensorFlow 2.21 on native Windows generally does not use CUDA GPU support. For GPU training, use a compatible Linux/WSL2 environment or Google Colab.

---

## 16. How to run the project

Run commands from:

```text
D:\DEEP LEARNING PROJECT\DEEP LEARNING PROJECT
```

### 16.1 Engine A

```powershell
& "D:\DEEP LEARNING PROJECT\.venv\Scripts\python.exe" `
  "D:\DEEP LEARNING PROJECT\DEEP LEARNING PROJECT\run_hybrid_nids.py" `
  --mode engine_a
```

Force retraining:

```powershell
& "D:\DEEP LEARNING PROJECT\.venv\Scripts\python.exe" `
  "D:\DEEP LEARNING PROJECT\DEEP LEARNING PROJECT\run_hybrid_nids.py" `
  --mode engine_a --force
```

### 16.2 Engine B

```powershell
& "D:\DEEP LEARNING PROJECT\.venv\Scripts\python.exe" `
  "D:\DEEP LEARNING PROJECT\DEEP LEARNING PROJECT\run_hybrid_nids.py" `
  --mode engine_b
```

### 16.3 Engine C for all days

```powershell
& "D:\DEEP LEARNING PROJECT\.venv\Scripts\python.exe" `
  "D:\DEEP LEARNING PROJECT\DEEP LEARNING PROJECT\run_hybrid_nids.py" `
  --mode engine_c
```

One day only:

```powershell
& "D:\DEEP LEARNING PROJECT\.venv\Scripts\python.exe" `
  "D:\DEEP LEARNING PROJECT\DEEP LEARNING PROJECT\run_hybrid_nids.py" `
  --mode engine_c --day Tuesday
```

### 16.4 Temporal evaluation

```powershell
& "D:\DEEP LEARNING PROJECT\.venv\Scripts\python.exe" `
  "D:\DEEP LEARNING PROJECT\DEEP LEARNING PROJECT\run_hybrid_nids.py" `
  --mode section5
```

### 16.5 Full pipeline

```powershell
& "D:\DEEP LEARNING PROJECT\.venv\Scripts\python.exe" `
  "D:\DEEP LEARNING PROJECT\DEEP LEARNING PROJECT\run_hybrid_nids.py" `
  --mode all
```

### 16.6 Hyperparameter tuning

Engine A:

```powershell
& "D:\DEEP LEARNING PROJECT\.venv\Scripts\python.exe" `
  "D:\DEEP LEARNING PROJECT\DEEP LEARNING PROJECT\tune_hyperparameters.py" `
  --engine a --trials 10
```

Engine C for each day:

```powershell
foreach ($day in @("Tuesday","Wednesday","Thursday","Friday")) {
    & "D:\DEEP LEARNING PROJECT\.venv\Scripts\python.exe" `
      "D:\DEEP LEARNING PROJECT\DEEP LEARNING PROJECT\tune_hyperparameters.py" `
      --engine c --day $day --trials 10
}
```

### 16.7 Dashboard

```powershell
Set-Location -LiteralPath "D:\DEEP LEARNING PROJECT\DEEP LEARNING PROJECT"

& ".venv\Scripts\python.exe" -m streamlit run dashboard\app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

---

## 17. Where to find results

### Engine A

`production_hybrid_nids/models/engine_a_flow_ae/`

`production_hybrid_nids/reports/engine_a_flow_ae_evaluation.json`

### Engine B

`production_hybrid_nids/clusters/`

### Engine C

`production_hybrid_nids/models/specialist_<Day>/`

`production_hybrid_nids/reports/`

### Temporal generalization

`production_hybrid_nids/reports/temporal_generalization_summary.csv`

`production_hybrid_nids/reports/temporal_hourly_drift.csv`

### Hyperparameter tuning

`tuning_results/`

### Dashboard

`dashboard/`

---

## 18. Limitations

1. CICIDS2017 is a benchmark dataset and may not represent every production network.
2. Engine A thresholds are calibrated from Monday benign traffic and may require recalibration for a different environment.
3. A high anomaly score does not automatically identify the exact attack type.
4. Engine B cluster labels are behavioral groups, not guaranteed ground-truth attack labels.
5. Rare classes have unstable percentage-based metrics because they contain very few examples.
6. The current saved Thursday and Friday specialist summary contains smoke-sized results and should be regenerated before publication.
7. Native Windows TensorFlow execution is CPU-based in the current environment.
8. The dashboard performs offline scoring rather than a complete packet-capture or streaming deployment.
9. The saved scikit-learn artifacts may produce version-compatibility warnings when loaded with a different scikit-learn version than the one used to create them.

---

## 19. Future work


- Retrain final models using the selected hyperparameters.
- Add a final untouched test evaluation after tuning.
- Add calibration curves and threshold optimization for operational false-positive targets.
- Add real-time packet or flow ingestion.
- Add model versioning and experiment tracking.
- Evaluate on additional datasets such as UNSW-NB15 or CIC-IDS2018.
- Add alert persistence, filtering, and export from the dashboard.

---

## 20. Viva explanation in short form

The project is a hybrid NIDS for both known and unseen attacks. The data is cleaned using leakage-safe preprocessing and reduced to behavioral features. Engine A is trained only on benign Monday traffic and detects unusual flows through autoencoder reconstruction error. Engine B clusters those suspicious flows using bottleneck embeddings, KMeans, and HDBSCAN. Engine C uses day-specific FT-Transformer and TCN classifiers to recognize known attacks. Finally, chronological evaluation demonstrates concept drift and shows why anomaly detection is needed in addition to supervised classification. The dashboard presents the saved models, anomaly scores, clusters, metrics, tuning results, and uploaded CSV predictions without retraining.

