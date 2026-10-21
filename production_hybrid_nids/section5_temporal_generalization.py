"""
Section 5: Intra-Day Temporal Generalization & Concept Drift Analysis
====================================================================
Evaluates the chronological generalization of intrusion detection models across
individual days in CICIDS2017.

Protocol:
1. Chronological Splitting:
   - First 70% of each day (chronologically sorted by timestamp) -> Train
   - Last 30% of each day -> Test
2. Zero Data Leakage:
   - StandardScaler and feature preprocessing are fit strictly on the first 70% train split.
3. Models Evaluated:
   - Deep Tabular FT-Transformer Classifier (Feature Tokenizer + Multi-Head Self-Attention).
4. Concept Drift & Temporal Degradation:
   - Compares Chronological F1 vs. Stratified Shuffle Split F1 to quantify drift magnitude (ΔF1).
   - Computes hourly performance metrics across the afternoon test period.
   - Analyzes regime changes (e.g., attacks ending or new attacks starting in the afternoon).
5. Visualizations & Reports:
   - Bar chart: Stratified vs. Chronological performance comparison.
   - Line chart: Hourly F1-score progression over time.
   - Summary CSVs and detailed JSON telemetry report.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Section5_Temporal")


@dataclass
class TemporalConfig:
    split_ratio: float = 0.70
    epochs: int = 5
    batch_size: int = 4096
    learning_rate: float = 2e-3
    patience: int = 3
    d_token: int = 32
    n_heads: int = 2
    dropout: float = 0.1
    max_class_weight: float = 250.0
    random_state: int = 42
    days: Tuple[str, ...] = ("Tuesday", "Wednesday", "Thursday", "Friday")


# =====================================================================
# FT-Transformer Tabular Architecture
# =====================================================================

class TabularFeatureTokenizer(layers.Layer):
    """Embeds each continuous tabular feature into a d_token dimensional vector."""
    def __init__(self, n_features: int, d_token: int, **kwargs):
        super().__init__(**kwargs)
        self.n_features = int(n_features)
        self.d_token = int(d_token)

    def build(self, shape):
        self.w = self.add_weight(
            shape=(self.n_features, self.d_token),
            initializer="glorot_uniform",
            name="feature_weights",
        )
        self.b = self.add_weight(
            shape=(self.n_features, self.d_token),
            initializer="zeros",
            name="feature_biases",
        )
        self.cls = self.add_weight(
            shape=(1, 1, self.d_token),
            initializer="glorot_uniform",
            name="cls_token",
        )

    def call(self, x):
        # x shape: (batch, n_features)
        tok = x[..., None] * self.w + self.b  # (batch, n_features, d_token)
        cls_tok = tf.tile(self.cls, [tf.shape(x)[0], 1, 1])  # (batch, 1, d_token)
        return tf.concat([cls_tok, tok], axis=1)  # (batch, n_features + 1, d_token)


def build_tabular_transformer(
    n_features: int,
    n_classes: int,
    config: TemporalConfig,
) -> keras.Model:
    """Builds a lightweight FT-Transformer classifier for tabular flow data."""
    inputs = layers.Input(shape=(n_features,), name="tabular_inputs")
    
    # 1. Feature Tokenization
    tokens = TabularFeatureTokenizer(n_features=n_features, d_token=config.d_token)(inputs)
    
    # 2. Multi-Head Self-Attention
    attn_out = layers.MultiHeadAttention(
        num_heads=config.n_heads,
        key_dim=config.d_token // config.n_heads,
        dropout=config.dropout,
    )(tokens, tokens)
    tokens = layers.LayerNormalization(epsilon=1e-5)(tokens + attn_out)
    
    # 3. Feed-Forward Network
    ff = layers.Dense(config.d_token * 2, activation="relu")(tokens)
    ff = layers.Dropout(config.dropout)(ff)
    ff = layers.Dense(config.d_token)(ff)
    tokens = layers.LayerNormalization(epsilon=1e-5)(tokens + ff)
    
    # 4. Classification Head on [CLS] token
    cls_token = tokens[:, 0, :]
    x = layers.Dense(64, activation="relu")(cls_token)
    x = layers.Dropout(config.dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="class_probs")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="ft_tabular_classifier")
    return model


# =====================================================================
# Temporal Generalization Engine
# =====================================================================

class TemporalGeneralizationEngine:
    def __init__(self, base_dir: Path, config: Optional[TemporalConfig] = None):
        self.base = Path(base_dir)
        self.cache_dir = self.base / "cache"
        self.feat_dir = self.base / "features"
        self.reports_dir = self.base / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or TemporalConfig()

        # Load selected feature list
        feat_path = self.feat_dir / "selected_features.json"
        if feat_path.exists():
            self.selected_features = json.loads(feat_path.read_text())["features"]
        else:
            self.selected_features = [f"feat_{i}" for i in range(32)]

        # Load Stratified Split baselines if available
        self.stratified_baselines = self._load_stratified_baselines()

    def _load_stratified_baselines(self) -> Dict[str, Dict[str, float]]:
        summary_path = self.reports_dir / "specialist_summary.csv"
        baselines = {}
        if summary_path.exists():
            try:
                df = pd.read_csv(summary_path)
                for _, row in df.iterrows():
                    d = str(row["day"])
                    baselines[d] = {
                        "weighted_f1": float(row.get("weighted_f1", np.nan)),
                        "ap": float(row.get("AP", np.nan)),
                        "roc_auc": float(row.get("ROC_AUC", np.nan)),
                    }
            except Exception as e:
                logger.warning(f"Could not load stratified summary: {e}")
        return baselines

    def run_day_experiment(self, day: str) -> Dict[str, Any]:
        """Runs the 70/30 chronological split experiment on a single day."""
        logger.info(f"\n{'='*70}\nRunning Temporal Generalization on {day}\n{'='*70}")
        
        # Load raw cleaned parquet for features and metadata parquet for timestamps & labels
        df_raw = pd.read_parquet(self.cache_dir / f"{day}.parquet")
        y_meta = pd.read_parquet(self.cache_dir / f"y_{day}.parquet")
        
        # Ensure chronological ordering
        if not y_meta["Timestamp"].is_monotonic_increasing:
            sort_idx = np.argsort(y_meta["Timestamp"].to_numpy())
            df_raw = df_raw.iloc[sort_idx].reset_index(drop=True)
            y_meta = y_meta.iloc[sort_idx].reset_index(drop=True)

        n_total = len(y_meta)
        split_idx = int(self.config.split_ratio * n_total)
        
        # Chronological Split
        t_split = y_meta.iloc[split_idx]["Timestamp"]
        logger.info(f"[{day}] Total flows: {n_total:,} | Split index: {split_idx:,} (70/30)")
        logger.info(f"[{day}] Train: {y_meta.iloc[0]['Timestamp']} to {t_split}")
        logger.info(f"[{day}] Test : {t_split} to {y_meta.iloc[-1]['Timestamp']}")

        meta_train = y_meta.iloc[:split_idx].copy()
        meta_test = y_meta.iloc[split_idx:].copy()

        # Select the 32 features
        available_feats = [f for f in self.selected_features if f in df_raw.columns]
        X_all = df_raw[available_feats].to_numpy().astype("float32")

        # Zero Data Leakage: Fit scaler strictly on the first 70%
        X_train_raw = X_all[:split_idx]
        X_test_raw = X_all[split_idx:]

        scaler = StandardScaler()
        # Handle infinities / extreme outliers with signed log1p before scaling
        X_train = scaler.fit_transform(np.sign(X_train_raw) * np.log1p(np.abs(X_train_raw)))
        X_test = scaler.transform(np.sign(X_test_raw) * np.log1p(np.abs(X_test_raw)))

        # Train label encoding
        train_labels = meta_train["Label"].to_numpy()
        test_labels = meta_test["Label"].to_numpy()

        known_classes = sorted(np.unique(train_labels).tolist())
        logger.info(f"[{day}] Classes in Chronological Train (first 70%): {dict(meta_train['Label'].value_counts())}")
        logger.info(f"[{day}] Classes in Chronological Test (last 30%) : {dict(meta_test['Label'].value_counts())}")

        le = LabelEncoder()
        le.fit(known_classes)
        y_train_enc = le.transform(train_labels)

        # For test evaluation, map unknown classes (not seen in morning) to -1
        y_test_enc = np.array([le.transform([lbl])[0] if lbl in le.classes_ else -1 for lbl in test_labels])

        # Compute balanced class weights for training
        classes_arr = np.unique(y_train_enc)
        weights = compute_class_weight("balanced", classes=classes_arr, y=y_train_enc)
        class_weight = {int(c): float(min(w, self.config.max_class_weight)) for c, w in zip(classes_arr, weights)}

        # Build and train Tabular FT-Transformer
        model = build_tabular_transformer(
            n_features=len(available_feats),
            n_classes=len(known_classes),
            config=self.config,
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Stratified holdout from morning train fold for validation
        from sklearn.model_selection import train_test_split
        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train_enc, test_size=0.10, random_state=self.config.random_state, stratify=y_train_enc
            )
        except Exception:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train_enc, test_size=0.10, random_state=self.config.random_state
            )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.patience,
                restore_best_weights=True,
                verbose=0,
            )
        ]

        logger.info(f"[{day}] Training FT-Transformer on chronological morning train split ({len(X_tr):,} train / {len(X_val):,} val)...")
        model.fit(
            X_tr,
            y_tr,
            validation_data=(X_val, y_val),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )

        # Predict on afternoon test split
        logger.info(f"[{day}] Evaluating on chronological afternoon test split ({len(X_test):,} flows)...")
        probs = model.predict(X_test, batch_size=4096, verbose=0)
        preds_enc = np.argmax(probs, axis=1)

        # -------------------------------------------------------------
        # 1. Binary Threat Detection Evaluation (Attack vs. Benign)
        # -------------------------------------------------------------
        y_test_bin = (test_labels != "BENIGN").astype(int)
        benign_idx = list(le.classes_).index("BENIGN") if "BENIGN" in le.classes_ else None
        
        if benign_idx is not None:
            attack_score = 1.0 - probs[:, benign_idx]
            pred_bin = (preds_enc != benign_idx).astype(int)
        else:
            attack_score = probs.max(axis=1)
            pred_bin = np.ones(len(preds_enc), dtype=int)

        bin_acc = float(accuracy_score(y_test_bin, pred_bin))
        bin_prec = float(precision_score(y_test_bin, pred_bin, zero_division=0))
        bin_rec = float(recall_score(y_test_bin, pred_bin, zero_division=0))
        bin_f1 = float(f1_score(y_test_bin, pred_bin, zero_division=0))
        
        has_both_classes = len(np.unique(y_test_bin)) == 2
        bin_ap = float(average_precision_score(y_test_bin, attack_score)) if has_both_classes else np.nan
        bin_auc = float(roc_auc_score(y_test_bin, attack_score)) if has_both_classes else np.nan

        # -------------------------------------------------------------
        # 2. Known Classes Evaluation (Excluding Unseen Novel Attacks)
        # -------------------------------------------------------------
        known_mask = y_test_enc != -1
        if known_mask.sum() > 0:
            known_y_true = y_test_enc[known_mask]
            known_y_pred = preds_enc[known_mask]
            known_weighted_f1 = float(f1_score(known_y_true, known_y_pred, average="weighted", zero_division=0))
            known_macro_f1 = float(f1_score(known_y_true, known_y_pred, average="macro", zero_division=0))
        else:
            known_weighted_f1 = np.nan
            known_macro_f1 = np.nan

        # Overall multiclass F1 (counting unlearned classes as missed)
        preds_labels = le.inverse_transform(preds_enc)
        overall_weighted_f1 = float(f1_score(test_labels, preds_labels, average="weighted", zero_division=0))

        # -------------------------------------------------------------
        # 3. Concept Drift & Comparison against Stratified Baseline
        # -------------------------------------------------------------
        strat_f1 = self.stratified_baselines.get(day, {}).get("weighted_f1", np.nan)
        strat_ap = self.stratified_baselines.get(day, {}).get("ap", np.nan)
        
        delta_f1 = (strat_f1 - overall_weighted_f1) if not np.isnan(strat_f1) else np.nan
        delta_ap = (strat_ap - bin_ap) if (not np.isnan(strat_ap) and not np.isnan(bin_ap)) else np.nan

        logger.info(f"[{day}] Chronological Overall Weighted F1 : {overall_weighted_f1:.4f}")
        logger.info(f"[{day}] Stratified Baseline Weighted F1    : {strat_f1:.4f}")
        logger.info(f"[{day}] Concept Drift Degradation (ΔF1)     : {delta_f1:.4f}")
        logger.info(f"[{day}] Binary Threat Detection Recall      : {bin_rec:.4f} (Prec: {bin_prec:.4f}, F1: {bin_f1:.4f})")

        # -------------------------------------------------------------
        # 4. Hourly Performance Breakdown (Drift Curve)
        # -------------------------------------------------------------
        meta_test["Hour"] = meta_test["Timestamp"].dt.hour
        meta_test["pred_bin"] = pred_bin
        meta_test["true_bin"] = y_test_bin

        hourly_records = []
        for hour, h_df in meta_test.groupby("Hour"):
            h_y_true = h_df["true_bin"].to_numpy()
            h_y_pred = h_df["pred_bin"].to_numpy()
            n_flows = len(h_df)
            n_attacks = int(h_y_true.sum())
            n_benign = n_flows - n_attacks
            
            h_rec = float(recall_score(h_y_true, h_y_pred, zero_division=0)) if n_attacks > 0 else 1.0
            h_prec = float(precision_score(h_y_true, h_y_pred, zero_division=0))
            h_f1 = float(f1_score(h_y_true, h_y_pred, zero_division=0))
            h_acc = float(accuracy_score(h_y_true, h_y_pred))

            hourly_records.append({
                "day": day,
                "hour": int(hour),
                "total_flows": int(n_flows),
                "benign_flows": int(n_benign),
                "attack_flows": int(n_attacks),
                "attack_ratio_pct": float(round((n_attacks / n_flows) * 100, 2)),
                "binary_accuracy": float(round(h_acc, 4)),
                "binary_precision": float(round(h_prec, 4)),
                "binary_recall": float(round(h_rec, 4)),
                "binary_f1": float(round(h_f1, 4)),
            })

        day_summary = {
            "day": day,
            "train_flows": int(split_idx),
            "test_flows": int(n_total - split_idx),
            "split_timestamp": str(t_split),
            "classes_in_train": known_classes,
            "classes_in_test": sorted(np.unique(test_labels).tolist()),
            "stratified_f1": strat_f1,
            "chronological_f1": overall_weighted_f1,
            "drift_delta_f1": delta_f1,
            "known_classes_f1": known_weighted_f1,
            "binary_threat_detection": {
                "accuracy": bin_acc,
                "precision": bin_prec,
                "recall": bin_rec,
                "f1": bin_f1,
                "ap": bin_ap,
                "roc_auc": bin_auc,
            },
            "hourly_breakdown": hourly_records,
        }
        return day_summary

    def run_all(self) -> Dict[str, Any]:
        """Executes Section 5 across all days and produces consolidated plots and reports."""
        all_day_summaries = {}
        all_hourly_records = []
        summary_rows = []

        for day in self.config.days:
            res = self.run_day_experiment(day)
            all_day_summaries[day] = res
            all_hourly_records.extend(res["hourly_breakdown"])

            b = res["binary_threat_detection"]
            summary_rows.append({
                "day": day,
                "train_flows": res["train_flows"],
                "test_flows": res["test_flows"],
                "split_time": res["split_timestamp"],
                "stratified_f1": round(res["stratified_f1"], 4) if not np.isnan(res["stratified_f1"]) else None,
                "chronological_f1": round(res["chronological_f1"], 4),
                "drift_delta_f1": round(res["drift_delta_f1"], 4) if not np.isnan(res["drift_delta_f1"]) else None,
                "binary_accuracy": round(b["accuracy"], 4),
                "binary_precision": round(b["precision"], 4),
                "binary_recall": round(b["recall"], 4),
                "binary_f1": round(b["f1"], 4),
                "binary_ap": round(b["ap"], 4) if not np.isnan(b["ap"]) else None,
                "binary_roc_auc": round(b["roc_auc"], 4) if not np.isnan(b["roc_auc"]) else None,
            })

        # Save CSV reports
        summary_df = pd.DataFrame(summary_rows)
        hourly_df = pd.DataFrame(all_hourly_records)

        summary_csv = self.reports_dir / "temporal_generalization_summary.csv"
        hourly_csv = self.reports_dir / "temporal_hourly_drift.csv"
        report_json = self.reports_dir / "temporal_concept_drift_report.json"

        summary_df.to_csv(summary_csv, index=False)
        hourly_df.to_csv(hourly_csv, index=False)
        report_json.write_text(json.dumps(all_day_summaries, indent=2, default=str))

        logger.info(f"Saved summary CSV to {summary_csv}")
        logger.info(f"Saved hourly CSV to {hourly_csv}")

        # Generate Visualizations
        self._plot_stratified_vs_chronological(summary_df)
        self._plot_hourly_f1_curve(hourly_df)

        return {
            "summary_table": summary_rows,
            "detailed": all_day_summaries,
        }

    def _plot_stratified_vs_chronological(self, df: pd.DataFrame):
        """Generates a comparison bar plot of Stratified vs. Chronological F1 score."""
        logger.info("Plotting Stratified vs Chronological performance comparison...")
        
        plot_df = []
        for _, r in df.iterrows():
            d = r["day"]
            if r["stratified_f1"] is not None:
                plot_df.append({"Day": d, "Split": "Stratified Shuffle (Random)", "F1": r["stratified_f1"]})
            plot_df.append({"Day": d, "Split": "Chronological (First 70% ➔ Last 30%)", "F1": r["chronological_f1"]})
        
        plot_df = pd.DataFrame(plot_df)

        plt.figure(figsize=(10, 5.5))
        palette = {"Stratified Shuffle (Random)": "#3b82f6", "Chronological (First 70% ➔ Last 30%)": "#ef4444"}
        ax = sns.barplot(data=plot_df, x="Day", y="F1", hue="Split", palette=palette)
        
        plt.title("Section 5: Stratified vs. Chronological F1-Score (Concept Drift Impact)", fontsize=13, fontweight="bold", pad=12)
        plt.ylabel("Weighted F1-Score", fontsize=11)
        plt.xlabel("Evaluation Day", fontsize=11)
        plt.ylim(0, 1.05)
        plt.grid(axis="y", alpha=0.3)
        plt.legend(title="Splitting Protocol", loc="lower left")

        # Annotate bars
        for p in ax.patches:
            height = p.get_height()
            if height > 0.05:
                ax.annotate(
                    f"{height:.1%}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    xytext=(0, 2),
                    textcoords="offset points",
                )

        plt.tight_layout()
        out_path = self.reports_dir / "stratified_vs_chronological_comparison.png"
        plt.savefig(out_path, dpi=140)
        plt.close()
        logger.info(f"Saved bar plot to {out_path}")

    def _plot_hourly_f1_curve(self, hourly_df: pd.DataFrame):
        """Plots the Hourly Binary F1-Score curve over the afternoon test hours."""
        logger.info("Plotting Hourly F1 curve...")
        
        plt.figure(figsize=(11, 5.5))
        
        # Filter days that have attack flows in the test period
        days_with_attacks = hourly_df[hourly_df["attack_flows"] > 0]["day"].unique()
        
        palette = sns.color_palette("tab10", n_colors=len(days_with_attacks))
        for i, d in enumerate(days_with_attacks):
            sub = hourly_df[hourly_df["day"] == d].sort_values("hour")
            plt.plot(
                sub["hour"],
                sub["binary_f1"],
                marker="o",
                linewidth=2.5,
                markersize=7,
                label=f"{d} (Afternoon Drift)",
                color=palette[i],
            )
            # Annotate flow counts
            for _, r in sub.iterrows():
                plt.annotate(
                    f"Attacks: {r['attack_flows']:,}",
                    (r["hour"], r["binary_f1"]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                    color="#475569",
                )

        plt.title("Intra-Day Temporal Generalization: Hourly Detection F1-Score Curve", fontsize=13, fontweight="bold", pad=12)
        plt.xlabel("Hour of Day (24-Hour Clock)", fontsize=11)
        plt.ylabel("Binary Attack Detection F1-Score", fontsize=11)
        plt.ylim(-0.05, 1.1)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower right")
        
        plt.tight_layout()
        out_path = self.reports_dir / "temporal_hourly_f1_curve.png"
        plt.savefig(out_path, dpi=140)
        plt.close()
        logger.info(f"Saved hourly curve plot to {out_path}")


if __name__ == "__main__":
    t0 = time.time()
    engine = TemporalGeneralizationEngine(base_dir=Path("production_hybrid_nids"))
    res = engine.run_all()
    print(f"\nSection 5 completed in {time.time() - t0:.1f}s.")
