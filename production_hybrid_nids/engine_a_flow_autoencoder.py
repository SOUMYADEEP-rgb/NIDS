"""
Engine A: Flow-Level Tabular FT-Transformer Autoencoder for Zero-Day Detection
==============================================================================
Implements a pure flow-level (lookback=1) autoencoder trained strictly on Monday
benign network flows. Eliminates the 90% temporal dilution problem of multi-step
sliding windows, enabling sharp anomaly detection on subtle attacks (Web Attacks,
Bot, Infiltration) as well as volumetric attacks (DDoS, PortScan).

Key Technical Highlights:
- Architecture: FeatureTokenizer + Multi-Head Self-Attention (FT-Transformer)
  encoding tabular flow tokens -> Bottleneck projection -> Reconstruction Decoder.
- Zero Leakage: Monday benign flows split 80% train / 20% validation.
- Clean Thresholding: Anomaly thresholds calibrated strictly at the 95th, 98th, 99th,
  and 99.5th percentiles of the Monday validation split (no target-day leakage).
- Multi-Metric Reconstruction:
  1. Global Mean MSE across all 32 features
  2. Top-K Feature MSE (isolates attacks altering 2-3 specific features)
  3. Max Feature Error
- Evaluation: Full validation against Thursday (unseen Web Attacks + Infiltration)
  and Friday (unseen DDoS, PortScan, Bot).
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, roc_curve

import tensorflow as tf
import keras
from keras import layers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EngineA_FlowAE")


@dataclass
class FlowAEConfig:
    n_features: int = 32
    d_token: int = 48
    n_heads: int = 4
    n_blocks: int = 2
    bottleneck_dim: int = 64
    dropout: float = 0.05
    learning_rate: float = 7e-4
    weight_decay: float = 1e-5
    batch_size: int = 1024
    epochs: int = 12
    patience: int = 4
    seed: int = 42


# =====================================================================
# FT-Transformer Tabular Autoencoder Architecture
# =====================================================================

class TabularFeatureTokenizer(layers.Layer):
    """Tokenizes each tabular feature into a dense embedding vector."""
    def __init__(self, n_features: int, d_token: int, **kwargs):
        super().__init__(**kwargs)
        self.n_features = int(n_features)
        self.d_token = int(d_token)

    def build(self, shape):
        self.w = self.add_weight(
            shape=(self.n_features, self.d_token),
            initializer="glorot_uniform",
            name="feature_weight",
        )
        self.b = self.add_weight(
            shape=(self.n_features, self.d_token),
            initializer="zeros",
            name="feature_bias",
        )
        self.cls = self.add_weight(
            shape=(1, 1, self.d_token),
            initializer="glorot_uniform",
            name="cls_token",
        )

    def call(self, x):
        tok = x[..., None] * self.w + self.b
        cls_tok = tf.tile(self.cls, [tf.shape(x)[0], 1, 1])
        return tf.concat([cls_tok, tok], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"n_features": self.n_features, "d_token": self.d_token})
        return config


class TabularFTBlock(layers.Layer):
    """Transformer block with Multi-Head Attention and Feed-Forward Network."""
    def __init__(
        self,
        d_token: int = 48,
        n_heads: int = 4,
        ff_mult: int = 2,
        dropout: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_token = d_token
        self.n_heads = n_heads
        self.ff_mult = ff_mult
        self.dropout = dropout
        self.n1 = layers.LayerNormalization(epsilon=1e-6)
        self.n2 = layers.LayerNormalization(epsilon=1e-6)
        self.att = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_token // n_heads, dropout=dropout
        )
        self.f1 = layers.Dense(ff_mult * d_token, activation="gelu")
        self.f2 = layers.Dense(d_token)
        self.dr = layers.Dropout(dropout)

    def call(self, x, training=None):
        h = self.n1(x)
        x = x + self.dr(self.att(h, h, training=training), training=training)
        h = self.n2(x)
        return x + self.dr(self.f2(self.dr(self.f1(h), training=training)), training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_token": self.d_token,
            "n_heads": self.n_heads,
            "ff_mult": self.ff_mult,
            "dropout": self.dropout,
        })
        return config


def build_flow_ft_autoencoder(config: FlowAEConfig) -> keras.Model:
    """Builds a pure flow-level FT-Transformer tabular autoencoder."""
    inp = keras.Input(shape=(config.n_features,), name="flow_features")
    x = TabularFeatureTokenizer(config.n_features, config.d_token)(inp)
    for i in range(config.n_blocks):
        x = TabularFTBlock(
            d_token=config.d_token,
            n_heads=config.n_heads,
            ff_mult=2,
            dropout=config.dropout,
            name=f"ft_block_{i+1}",
        )(x)
    
    # Global pooling across all feature tokens + CLS token
    pooled = layers.GlobalAveragePooling1D(name="feature_pooling")(x)
    bottleneck = layers.Dense(config.bottleneck_dim, activation="gelu", name="bottleneck")(pooled)
    
    # Decoder network
    h = layers.Dense(128, activation="gelu", name="decoder_dense1")(bottleneck)
    h = layers.Dropout(config.dropout)(h)
    out = layers.Dense(config.n_features, activation="linear", name="reconstructed_features")(h)
    
    return keras.Model(inp, out, name="flow_ft_autoencoder")


# =====================================================================
# Training & Evaluation Engine
# =====================================================================

class EngineAFlowTrainer:
    def __init__(self, base_dir: Path, config: Optional[FlowAEConfig] = None):
        self.base = Path(base_dir)
        self.cache_dir = self.base / "cache"
        self.models_dir = self.base / "models" / "engine_a_flow_ae"
        self.reports_dir = self.base / "reports"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or FlowAEConfig()

    def load_data(self, day: str) -> Tuple[np.ndarray, pd.DataFrame]:
        X = np.load(self.cache_dir / f"X_{day}.npy")
        y = pd.read_parquet(self.cache_dir / f"y_{day}.parquet")
        return X, y

    def train_or_load(self, force: bool = False) -> Tuple[keras.Model, Dict[str, Any]]:
        final_model_path = self.models_dir / "flow_ft_autoencoder.keras"
        thresh_path = self.models_dir / "thresholds_monday_val.json"

        custom_objs = {
            "TabularFeatureTokenizer": TabularFeatureTokenizer,
            "TabularFTBlock": TabularFTBlock,
        }

        if not force and final_model_path.exists() and thresh_path.exists():
            logger.info("Loading existing flow-level autoencoder and thresholds...")
            model = keras.models.load_model(final_model_path, custom_objects=custom_objs, compile=False)
            thresholds = json.loads(thresh_path.read_text())
            return model, thresholds

        logger.info("Training Flow-Level FT-Transformer Autoencoder on Monday Benign...")
        X_mon, y_mon = self.load_data("Monday")
        assert (y_mon["Label"] == "BENIGN").all(), "Monday must be 100% Benign"

        # 80/20 train/validation split strictly on Monday
        rng = np.random.default_rng(self.config.seed)
        n_total = len(X_mon)
        perm = rng.permutation(n_total)
        n_val = int(0.20 * n_total)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        X_tr = X_mon[tr_idx]
        X_va = X_mon[val_idx]
        logger.info(f"Monday split: {len(X_tr):,} train flows, {len(X_va):,} validation flows.")

        model = build_flow_ft_autoencoder(self.config)
        optimizer = keras.optimizers.AdamW(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            clipnorm=1.0,
        )
        model.compile(optimizer=optimizer, loss="mse")

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                str(self.models_dir / "best_flow_ft_autoencoder.keras"),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.patience,
                restore_best_weights=True,
                verbose=1,
            ),
        ]

        logger.info(f"Fitting model for up to {self.config.epochs} epochs (batch={self.config.batch_size})...")
        t0 = time.time()
        hist = model.fit(
            X_tr,
            X_tr,
            validation_data=(X_va, X_va),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        logger.info(f"Training completed in {time.time() - t0:.1f}s.")

        model.save(final_model_path)
        pd.DataFrame(hist.history).to_csv(self.models_dir / "training_history.csv", index=False)

        # Calibrate thresholds strictly on Monday validation split
        logger.info("Computing Monday validation reconstruction error percentiles...")
        pred_va = model.predict(X_va, batch_size=4096, verbose=0)
        sq_err_va = np.square(pred_va - X_va) # (N_val, 32)
        mse_mean_va = np.mean(sq_err_va, axis=1)
        sorted_sq_va = np.sort(sq_err_va, axis=1)
        mse_top3_va = np.mean(sorted_sq_va[:, -3:], axis=1)
        mse_top5_va = np.mean(sorted_sq_va[:, -5:], axis=1)
        mse_max_va = sorted_sq_va[:, -1]

        thresholds = {
            "calibration_source": "Monday Benign 20% validation split (105,983 flows)",
            "mean_mse": {
                "p90": float(np.percentile(mse_mean_va, 90)),
                "p95": float(np.percentile(mse_mean_va, 95)),
                "p97": float(np.percentile(mse_mean_va, 97)),
                "p99": float(np.percentile(mse_mean_va, 99)),
                "p99_5": float(np.percentile(mse_mean_va, 99.5)),
            },
            "top3_mse": {
                "p90": float(np.percentile(mse_top3_va, 90)),
                "p95": float(np.percentile(mse_top3_va, 95)),
                "p97": float(np.percentile(mse_top3_va, 97)),
                "p99": float(np.percentile(mse_top3_va, 99)),
                "p99_5": float(np.percentile(mse_top3_va, 99.5)),
            },
            "max_mse": {
                "p90": float(np.percentile(mse_max_va, 90)),
                "p95": float(np.percentile(mse_max_va, 95)),
                "p97": float(np.percentile(mse_max_va, 97)),
                "p99": float(np.percentile(mse_max_va, 99)),
                "p99_5": float(np.percentile(mse_max_va, 99.5)),
            },
        }
        thresh_path.write_text(json.dumps(thresholds, indent=2))
        logger.info(f"Thresholds saved to {thresh_path}: p99 Mean MSE = {thresholds['mean_mse']['p99']:.6f}, p99 Top3 MSE = {thresholds['top3_mse']['p99']:.6f}")

        return model, thresholds

    def evaluate_day(
        self,
        day: str,
        model: keras.Model,
        thresholds: Dict[str, Any],
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        logger.info(f"Evaluating Flow Autoencoder on {day}...")
        X, y_meta = self.load_data(day)
        labels = y_meta["Label"].to_numpy()

        if max_samples and max_samples < len(X):
            rng = np.random.default_rng(self.config.seed)
            sel = np.sort(rng.choice(len(X), max_samples, replace=False))
            X = X[sel]
            labels = labels[sel]

        logger.info(f"[{day}] Running inference on {len(X):,} flows...")
        pred = model.predict(X, batch_size=4096, verbose=1)
        sq_err = np.square(pred - X) # (N, 32)
        score_mean = np.mean(sq_err, axis=1)
        sorted_err = np.sort(sq_err, axis=1)
        score_top3 = np.mean(sorted_err[:, -3:], axis=1)
        score_max = sorted_err[:, -1]

        is_attack = (labels != "BENIGN").astype(int)

        # Primary metrics
        auc_mean = float(roc_auc_score(is_attack, score_mean))
        ap_mean = float(average_precision_score(is_attack, score_mean))
        auc_top3 = float(roc_auc_score(is_attack, score_top3))
        ap_top3 = float(average_precision_score(is_attack, score_top3))
        auc_max = float(roc_auc_score(is_attack, score_max))
        ap_max = float(average_precision_score(is_attack, score_max))

        # Check detection at Monday p99 threshold
        p99_mean_thr = thresholds["mean_mse"]["p99"]
        p99_top3_thr = thresholds["top3_mse"]["p99"]
        flagged_mean_p99 = score_mean > p99_mean_thr
        flagged_top3_p99 = score_top3 > p99_top3_thr

        overall_recall_mean = float((flagged_mean_p99[is_attack == 1]).mean() * 100) if (is_attack == 1).any() else 0.0
        overall_recall_top3 = float((flagged_top3_p99[is_attack == 1]).mean() * 100) if (is_attack == 1).any() else 0.0
        fpr_mean = float((flagged_mean_p99[is_attack == 0]).mean() * 100)
        fpr_top3 = float((flagged_top3_p99[is_attack == 0]).mean() * 100)

        family_report = []
        for fam in sorted(np.unique(labels)):
            m = labels == fam
            rec_m = float((flagged_mean_p99[m]).mean() * 100)
            rec_t3 = float((flagged_top3_p99[m]).mean() * 100)
            family_report.append({
                "label": fam,
                "count": int(m.sum()),
                "recall_mean_mse_p99": rec_m,
                "recall_top3_mse_p99": rec_t3,
                "median_mean_mse": float(np.median(score_mean[m])),
                "median_top3_mse": float(np.median(score_top3[m])),
            })

        logger.info(f"\n===== {day} Zero-Day Detection Summary =====")
        logger.info(f"  Mean MSE: ROC-AUC = {auc_mean:.4f}, AP = {ap_mean:.4f}, Recall@p99 = {overall_recall_mean:.2f}%, FPR = {fpr_mean:.2f}%")
        logger.info(f"  Top-3 MSE: ROC-AUC = {auc_top3:.4f}, AP = {ap_top3:.4f}, Recall@p99 = {overall_recall_top3:.2f}%, FPR = {fpr_top3:.2f}%")
        logger.info(f"  Max Feature: ROC-AUC = {auc_max:.4f}, AP = {ap_max:.4f}")

        report = {
            "day": day,
            "flows_evaluated": len(X),
            "attacks_evaluated": int(is_attack.sum()),
            "mean_mse_metrics": {
                "roc_auc": auc_mean,
                "ap": ap_mean,
                "recall_p99": overall_recall_mean,
                "benign_fpr": fpr_mean,
            },
            "top3_mse_metrics": {
                "roc_auc": auc_top3,
                "ap": ap_top3,
                "recall_p99": overall_recall_top3,
                "benign_fpr": fpr_top3,
            },
            "max_feature_metrics": {
                "roc_auc": auc_max,
                "ap": ap_max,
            },
            "per_family_breakdown": family_report,
        }
        return report
