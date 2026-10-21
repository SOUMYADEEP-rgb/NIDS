"""Viva-focused, read-only dashboard for the Hybrid NIDS project.

Run from the project root with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "production_hybrid_nids"
REPORTS = BASE / "reports"
MODELS = BASE / "models"
FEATURES = BASE / "features"
CLUSTERS = BASE / "clusters"
TUNING = ROOT / "tuning_results"
sys.path.insert(0, str(ROOT))


st.set_page_config(page_title="Hybrid NIDS Dashboard", page_icon="🛡️", layout="wide")


def read_json(path: Path, default: dict | list | None = None):
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default


def safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_resource(show_spinner="Loading Engine A model...")
def load_engine_a():
    from tensorflow import keras

    from production_hybrid_nids.engine_a_flow_autoencoder import (
        TabularFeatureTokenizer,
        TabularFTBlock,
    )

    path = MODELS / "engine_a_flow_ae" / "best_flow_ft_autoencoder.keras"
    if not path.exists():
        raise FileNotFoundError(f"Missing Engine A model: {path}")
    model = keras.models.load_model(
        path,
        custom_objects={
            "TabularFeatureTokenizer": TabularFeatureTokenizer,
            "TabularFTBlock": TabularFTBlock,
        },
        compile=False,
    )
    thresholds = read_json(MODELS / "engine_a_flow_ae" / "thresholds_monday_val.json")
    return model, thresholds


def slog(values: np.ndarray) -> np.ndarray:
    return (np.sign(values) * np.log1p(np.abs(values))).astype("float32")


@st.cache_data
def engine_a_features() -> tuple[list[str], pd.Series, object]:
    selected_path = FEATURES / "selected_features.json"
    selected_data = read_json(selected_path)
    selected = selected_data.get("features", selected_data if isinstance(selected_data, list) else [])
    if not selected:
        raise FileNotFoundError(f"No selected features found in {selected_path}")

    medians_path = FEATURES / "train_medians.joblib"
    medians = joblib.load(medians_path) if medians_path.exists() else pd.Series(dtype="float32")
    if isinstance(medians, dict):
        medians = medians.get("medians", medians)
    scaler = joblib.load(FEATURES / "scaler.joblib")
    return list(selected), medians, scaler


def score_uploaded_csv(uploaded: bytes) -> tuple[pd.DataFrame, dict]:
    import io

    frame = pd.read_csv(io.BytesIO(uploaded))
    selected, medians, scaler = engine_a_features()
    missing = [name for name in selected if name not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing {len(missing)} required model features: {missing[:8]}")

    values = frame[selected].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).fillna(medians)
    transformed = np.clip(scaler.transform(slog(values.to_numpy("float32"))), -5.0, 5.0).astype("float32")
    model, thresholds = load_engine_a()
    reconstructed = model.predict(transformed, batch_size=2048, verbose=0)
    squared_error = np.square(reconstructed - transformed)
    mean_mse = squared_error.mean(axis=1)
    top3_mse = np.sort(squared_error, axis=1)[:, -3:].mean(axis=1)
    max_mse = squared_error.max(axis=1)
    mean_threshold = thresholds.get("mean_mse", {}).get("p95", float(np.quantile(mean_mse, 0.95)))
    top3_threshold = thresholds.get("top3_mse", {}).get("p95", float(np.quantile(top3_mse, 0.95)))
    result = frame.copy()
    result["mean_mse"] = mean_mse
    result["top3_mse"] = top3_mse
    result["max_mse"] = max_mse
    result["is_anomaly"] = (mean_mse > mean_threshold) | (top3_mse > top3_threshold)
    metadata = {
        "rows": len(result),
        "anomalies": int(result["is_anomaly"].sum()),
        "mean_threshold": mean_threshold,
        "top3_threshold": top3_threshold,
    }
    return result, metadata


def overview_page() -> None:
    st.title("🛡️ Hybrid NIDS Dashboard")
    st.caption("Read-only project overview: existing artifacts and reports are displayed; models are not retrained.")
    audit = safe_read_csv(REPORTS / "cleaning_audit.csv")
    cluster = read_json(CLUSTERS / "cluster_evaluation.json")
    total_flows = int(audit.get("rows_clean", pd.Series(dtype=int)).sum()) if not audit.empty else 0
    cluster_count = cluster.get("n_clusters", "—")
    model_count = len(list(MODELS.glob("**/*.keras")))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cleaned flows", f"{total_flows:,}" if total_flows else "Available in reports")
    c2.metric("Saved Keras models", model_count)
    c3.metric("KMeans clusters", cluster_count)
    c4.metric("Silhouette score", cluster.get("metrics", {}).get("silhouette_score", "—"))
    st.subheader("Pipeline")
    st.info("Raw flow → preprocessing and 32 selected features → Engine A anomaly detection → Engine B clustering → Engine C known-attack classification")
    st.subheader("Artifact status")
    status = pd.DataFrame({
        "Component": ["Engine A model", "Engine A thresholds", "Engine B evaluation", "Engine C summary", "Tuning results"],
        "Path": [
            MODELS / "engine_a_flow_ae" / "best_flow_ft_autoencoder.keras",
            MODELS / "engine_a_flow_ae" / "thresholds_monday_val.json",
            CLUSTERS / "cluster_evaluation.json",
            REPORTS / "specialist_summary.csv",
            TUNING,
        ],
    })
    status["Available"] = status["Path"].map(Path.exists)
    st.dataframe(status.drop(columns="Path"), use_container_width=True, hide_index=True)


def prediction_page() -> None:
    st.title("📄 CSV Upload and Engine A Prediction")
    st.caption("Upload a CICIDS-style CSV containing the selected numeric flow features.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to calculate reconstruction errors and anomaly flags.")
        return
    if st.button("Score with Engine A", type="primary"):
        try:
            result, metadata = score_uploaded_csv(uploaded.getvalue())
            st.session_state["prediction_result"] = result
            st.session_state["prediction_metadata"] = metadata
        except Exception as exc:  # dashboard should show an actionable message
            st.error(str(exc))
    result = st.session_state.get("prediction_result")
    metadata = st.session_state.get("prediction_metadata")
    if result is None or metadata is None:
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows scored", f"{metadata['rows']:,}")
    c2.metric("Anomalies", f"{metadata['anomalies']:,}")
    c3.metric("Anomaly rate", f"{metadata['anomalies'] / max(metadata['rows'], 1) * 100:.2f}%")
    st.write(f"Mean-MSE p95 threshold: `{metadata['mean_threshold']:.6f}`; Top-3-MSE p95 threshold: `{metadata['top3_threshold']:.6f}`")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(result["mean_mse"], bins=40, color="#2563eb", alpha=0.8)
    ax.axvline(metadata["mean_threshold"], color="#dc2626", linestyle="--", label="p95 threshold")
    ax.set_title("Engine A mean reconstruction error")
    ax.set_xlabel("Mean MSE")
    ax.set_ylabel("Flow count")
    ax.legend()
    st.pyplot(fig, clear_figure=True)
    st.dataframe(result[["mean_mse", "top3_mse", "max_mse", "is_anomaly"]].head(500), use_container_width=True)


def engine_b_page() -> None:
    st.title("🧩 Engine B: Anomaly Clusters")
    evaluation = read_json(CLUSTERS / "cluster_evaluation.json")
    metrics = evaluation.get("metrics", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Silhouette", metrics.get("silhouette_score", "—"))
    c2.metric("Calinski–Harabasz", metrics.get("calinski_harabasz_index", "—"))
    c3.metric("Davies–Bouldin", metrics.get("davies_bouldin_index", "—"))
    image = CLUSTERS / "cluster_manifold_projection.png"
    if image.exists():
        st.image(str(image), caption="2D projection of Engine A anomaly embeddings", use_container_width=True)
    audit = evaluation.get("cluster_audit", [])
    if audit:
        st.dataframe(pd.json_normalize(audit), use_container_width=True, hide_index=True)
    st.caption("Cluster labels shown in the audit are for post-hoc analysis; Engine B itself is unsupervised.")


def engine_c_page() -> None:
    st.title("🎯 Engine C: Day-wise Classification")
    summary = safe_read_csv(REPORTS / "specialist_summary.csv")
    if summary.empty:
        st.warning("specialist_summary.csv was not found.")
        return
    st.dataframe(summary, use_container_width=True, hide_index=True)
    day_options = ["Tuesday", "Wednesday", "Thursday", "Friday"]
    day = st.selectbox("View day report", day_options)
    metric_path = REPORTS / f"specialist_{day}_metrics.json"
    metrics = read_json(metric_path)
    if metrics:
        st.json(metrics)
    for suffix in ["confusion_matrix.png", "roc_pr.png"]:
        image = REPORTS / f"specialist_{day}_{suffix}"
        if image.exists():
            st.image(str(image), caption=f"{day} {suffix.replace('_', ' ').replace('.png', '')}", use_container_width=True)


def tuning_page() -> None:
    st.title("⚙️ Hyperparameter Tuning")
    files = sorted(TUNING.glob("*_trials.csv"))
    if not files:
        st.info("No tuning CSV files found yet. Run tune_hyperparameters.py first.")
        return
    selected = st.selectbox("Trial file", files, format_func=lambda path: path.name)
    trials = pd.read_csv(selected)
    st.dataframe(trials, use_container_width=True, hide_index=True)
    best = selected.with_name(selected.name.replace("_trials.csv", "_best_params.json"))
    if best.exists():
        st.subheader("Best configuration")
        st.json(read_json(best))


PAGES = {
    "Overview": overview_page,
    "CSV prediction": prediction_page,
    "Engine A": prediction_page,
    "Engine B": engine_b_page,
    "Engine C": engine_c_page,
    "Hyperparameter tuning": tuning_page,
}
page = st.sidebar.radio("Navigate", list(PAGES))
st.sidebar.caption("No training is started by this dashboard.")
PAGES[page]()
