"""
Engine C: Isolated Per-Day Specialist Classifiers
=================================================
Implements modular supervised hybrid FT-Transformer + TCN classifiers for each
distinct day in CICIDS2017 with attacks (Tuesday, Wednesday, Thursday, Friday).

Key Architectural & Evaluation Principles:
- Complete isolation: Each day trains independently on its own traffic distribution.
- Zero data leakage: 70/30 StratifiedShuffleSplit happens BEFORE medians, scaling,
  or feature selection. Only train-split statistics are used.
- Feature selection: Train-fold dual ranking using Random Forest importance + Mutual Information.
- Hybrid Architecture: Feature Tokenizer + Multi-Head Self-Attention (FT-Transformer)
  feeding into Temporal Convolutional Networks (TCN) with dilated causal convolutions.
- Class Imbalance: Balanced class weighting with adaptive cap to provide gradient signal
  for rare classes (e.g. Infiltration, SQL Injection, Heartbleed).
- Comprehensive Validation: Precision-Recall curves, ROC curves, confusion matrix heatmaps,
  per-class classification reports, and cross-day summary scorecards.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EngineC")


@dataclass
class EngineCConfig:
    """Configuration for Engine C Day Specialists."""
    lookback: int = 10
    batch_size: int = 512
    epochs: int = 20
    patience: int = 5
    top_k_features: int = 32
    d_token: int = 48
    n_heads: int = 4
    n_ft_blocks: int = 2
    tcn_filters: int = 96
    tcn_kernel_size: int = 3
    tcn_dilations: Tuple[int, ...] = (1, 2, 4, 8)
    dropout: float = 0.1
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    max_class_weight: float = 250.0
    seed: int = 42
    smoke_test: bool = False
    smoke_samples: int = 5000


def slog(a: np.ndarray) -> np.ndarray:
    """Safe signed logarithm transform: sign(x) * log1p(|x|)."""
    return (np.sign(a) * np.log1p(np.abs(a))).astype("float32")


# =====================================================================
# FT-Transformer & TCN Model Components
# =====================================================================

class FeatureTokenizer(layers.Layer):
    """Tokenizes each continuous tabular feature into an embedding vector."""
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


class FTBlock(layers.Layer):
    """Transformer block with Multi-Head Attention and Feed-Forward Network."""
    def __init__(
        self,
        d_token: int = 48,
        n_heads: int = 4,
        ff_mult: int = 2,
        dropout: float = 0.1,
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


def build_ft_encoder(
    n_features: int,
    d_token: int = 48,
    n_heads: int = 4,
    n_blocks: int = 2,
    dropout: float = 0.1,
    name: str = "cls_ft_encoder",
) -> keras.Model:
    """Builds the FT-Transformer encoder mapping a single flow vector to an embedding."""
    inp = keras.Input((n_features,), name="flow_features")
    x = FeatureTokenizer(n_features, d_token)(inp)
    for i in range(n_blocks):
        x = FTBlock(d_token, n_heads, 2, dropout, name=f"ft_block_{i+1}")(x)
    cls_repr = layers.LayerNormalization(epsilon=1e-6)(x[:, 0, :])
    return keras.Model(inp, cls_repr, name=name)


def tcn_stack(
    x,
    filters: int = 96,
    kernel_size: int = 3,
    dilations: Tuple[int, ...] = (1, 2, 4, 8),
    dropout: float = 0.1,
):
    """Stack of Dilated Causal Convolutional blocks with residual shortcuts."""
    for d in dilations:
        res = x
        h = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=d)(x)
        h = layers.LayerNormalization(epsilon=1e-6)(h)
        h = layers.Activation("relu")(h)
        h = layers.SpatialDropout1D(dropout)(h)
        h = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=d)(h)
        h = layers.LayerNormalization(epsilon=1e-6)(h)
        if res.shape[-1] != filters:
            res = layers.Conv1D(filters, 1, padding="same")(res)
        x = layers.Activation("relu")(layers.Add()([h, res]))
    return x


def build_specialist_model(
    n_features: int,
    lookback: int,
    n_classes: int,
    config: EngineCConfig,
    name: str = "ft_transformer_tcn_specialist",
) -> keras.Model:
    """Builds the complete hybrid FT-Transformer + TCN specialist classifier."""
    inp = keras.Input((lookback, n_features), name="window_features")
    ft_enc = build_ft_encoder(
        n_features,
        d_token=config.d_token,
        n_heads=config.n_heads,
        n_blocks=config.n_ft_blocks,
        dropout=config.dropout,
        name="cls_ft_encoder",
    )
    # Tokenize and encode each flow in the temporal window
    z = layers.TimeDistributed(ft_enc, name="encode_each_flow")(inp)
    # Model temporal dependencies across the window
    z = tcn_stack(
        z,
        filters=config.tcn_filters,
        kernel_size=config.tcn_kernel_size,
        dilations=config.tcn_dilations,
        dropout=config.dropout,
    )
    z = layers.GlobalAveragePooling1D(name="temporal_pooling")(z)
    z = layers.Dense(128, activation="gelu", name="dense_proj")(z)
    z = layers.Dropout(0.2, name="head_dropout")(z)
    out = layers.Dense(n_classes, activation="softmax", name="class_probability")(z)
    return keras.Model(inp, out, name=name)


def make_windows(
    X: np.ndarray, labels: Optional[np.ndarray] = None, lookback: int = 10, stride: int = 1
) -> Tuple[np.ndarray, ...]:
    """Generates sliding temporal windows: (N - lookback + 1, lookback, n_features)."""
    ends = np.arange(lookback - 1, len(X), stride, dtype=np.int64)
    win = np.stack([X[e - lookback + 1 : e + 1] for e in ends]).astype("float32")
    if labels is None:
        return win, ends
    return win, labels[ends], ends


# =====================================================================
# Engine C Pipeline & Trainer
# =====================================================================

class SpecialistEngine:
    """Orchestrator for training and evaluating Day Specialist classifiers."""

    def __init__(self, base_dir: Path, config: Optional[EngineCConfig] = None):
        self.base = Path(base_dir)
        self.cache_dir = self.base / "cache"
        self.feat_dir = self.base / "features"
        self.models_dir = self.base / "models"
        self.reports_dir = self.base / "reports"
        self.config = config or EngineCConfig()

        for d in [self.cache_dir, self.feat_dir, self.models_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

        raw_feat_path = self.feat_dir / "raw_feature_candidates.json"
        if raw_feat_path.exists():
            self.raw_features = json.loads(raw_feat_path.read_text())
        else:
            raise FileNotFoundError(f"Missing {raw_feat_path}")

    def load_day(self, day: str) -> pd.DataFrame:
        """Loads clean day parquet from cache."""
        pq = self.cache_dir / f"{day}.parquet"
        if not pq.exists():
            raise FileNotFoundError(f"Missing {pq}. Rebuild clean cache first.")
        return pd.read_parquet(pq)

    def stratified_day_split(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Performs a 70/30 StratifiedShuffleSplit handling ultra-rare attack classes."""
        y = df["Label"].to_numpy()
        vc = pd.Series(y).value_counts()
        strat = np.array([
            lbl if vc[lbl] >= 2 else ("RARE_ATTACK" if lbl != "BENIGN" else "BENIGN")
            for lbl in y
        ])
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=self.config.seed)
        train_idx, test_idx = next(splitter.split(np.zeros(len(strat)), strat))
        return np.sort(train_idx), np.sort(test_idx)

    def fit_day_pipeline(
        self, day: str, df: pd.DataFrame, train_idx: np.ndarray
    ) -> Tuple[List[str], pd.Series, StandardScaler]:
        """Fits train-split-only medians, RF+MI ranking, and scaler (Zero Data Leakage)."""
        fdir = self.feat_dir / f"specialist_{day}"
        fdir.mkdir(parents=True, exist_ok=True)

        tr = df.iloc[train_idx].reset_index(drop=True)
        med = tr[self.raw_features].median()
        imp = tr[self.raw_features].fillna(med)

        const = imp.nunique()[lambda s: s <= 1].index.tolist()
        rem = [c for c in self.raw_features if c not in const]

        y_bin = (tr["Label"] != "BENIGN").astype("int8").to_numpy()
        rng = np.random.default_rng(self.config.seed)

        def cap(y_arr, n_samples):
            parts = []
            for c in np.unique(y_arr):
                idx = np.where(y_arr == c)[0]
                n_pick = min(n_samples, len(idx))
                parts.append(rng.choice(idx, n_pick, replace=False))
            return np.sort(np.concatenate(parts))

        X = imp[rem].to_numpy("float32")
        irf = cap(y_bin, 60_000)
        imi = cap(y_bin, 12_000)

        logger.info(f"[{day}] Ranking features with Random Forest (160 trees)...")
        rf = RandomForestClassifier(
            n_estimators=160,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=self.config.seed,
        ).fit(X[irf], y_bin[irf])

        logger.info(f"[{day}] Computing Mutual Information scores...")
        mi = mutual_info_classif(X[imi], y_bin[imi], random_state=self.config.seed)

        rank = pd.DataFrame({
            "feature": rem,
            "rf_importance": rf.feature_importances_,
            "mutual_info": mi,
        })
        rank["combined_rank"] = (
            rank["rf_importance"].rank(ascending=False)
            + rank["mutual_info"].rank(ascending=False)
        ) / 2
        rank = rank.sort_values(
            ["combined_rank", "rf_importance"], ascending=[True, False]
        ).reset_index(drop=True)

        selected_feats = rank["feature"].head(min(self.config.top_k_features, len(rank))).tolist()

        logger.info(f"[{day}] Fitting StandardScaler on training split ({len(selected_feats)} features)...")
        scaler = StandardScaler().fit(slog(tr[selected_feats].fillna(med[selected_feats]).to_numpy("float32")))

        # Persist pipeline artifacts
        rank.to_csv(fdir / "feature_ranking.csv", index=False)
        (fdir / "selected_features.json").write_text(
            json.dumps({
                "day": day,
                "k": len(selected_feats),
                "features": selected_feats,
                "train_split_only": True,
            }, indent=2)
        )
        joblib.dump({"medians": med[selected_feats], "scaler": scaler, "rf_selector": rf}, fdir / "pipeline.joblib")

        return selected_feats, med[selected_feats], scaler

    def transform_day(
        self, df: pd.DataFrame, feats: List[str], med: pd.Series, scaler: StandardScaler
    ) -> np.ndarray:
        """Transforms a DataFrame using the frozen train-split pipeline."""
        raw_mat = df[feats].fillna(med).to_numpy("float32")
        scaled = scaler.transform(slog(raw_mat))
        return np.clip(scaled, -5.0, 5.0).astype("float32")

    def train_specialist(self, day: str, force: bool = False) -> Dict[str, Any]:
        """Instantiates, trains, and comprehensively evaluates a day specialist."""
        mdir = self.models_dir / f"specialist_{day}"
        mdir.mkdir(parents=True, exist_ok=True)
        final_model_path = mdir / f"specialist_{day}.keras"
        metrics_path = self.reports_dir / f"specialist_{day}_metrics.json"

        if not force and final_model_path.exists() and metrics_path.exists():
            logger.info(f"[{day}] Specialist already trained and evaluated. Loading metrics.")
            metrics = json.loads(metrics_path.read_text())
            # Ensure plots exist even if previously trained
            self._ensure_plots(day, mdir, metrics)
            return metrics

        logger.info(f"========== Training Specialist for {day} ==========")
        df = self.load_day(day)
        logger.info(f"[{day}] Loaded {len(df):,} flows. Labels: {dict(df['Label'].value_counts())}")

        tr_idx, te_idx = self.stratified_day_split(df)
        logger.info(f"[{day}] Split: {len(tr_idx):,} train flows, {len(te_idx):,} test flows (70/30).")

        feats, med, scaler = self.fit_day_pipeline(day, df, tr_idx)
        X = self.transform_day(df, feats, med, scaler)

        le = LabelEncoder().fit(df["Label"].iloc[tr_idx])
        classes = np.arange(len(le.classes_))
        labels = list(le.classes_)

        # Filter test indices in case of any unobserved label in train
        te_idx = te_idx[np.isin(df["Label"].iloc[te_idx], le.classes_)]
        y_train = le.transform(df["Label"].iloc[tr_idx])
        y_test = le.transform(df["Label"].iloc[te_idx])

        logger.info(f"[{day}] Creating sliding temporal windows (lookback={self.config.lookback})...")
        Xtr, ytr, _ = make_windows(X[tr_idx], y_train, self.config.lookback)
        Xte, yte, _ = make_windows(X[te_idx], y_test, self.config.lookback)

        if self.config.smoke_test:
            logger.warning(f"[{day}] SMOKE TEST ENABLED: Subsampling {self.config.smoke_samples} windows.")
            N = self.config.smoke_samples
            rng = np.random.default_rng(self.config.seed)
            idx_tr = np.sort(rng.choice(len(Xtr), min(N, len(Xtr)), replace=False))
            idx_te = np.sort(rng.choice(len(Xte), min(N, len(Xte)), replace=False))
            Xtr, ytr = Xtr[idx_tr], ytr[idx_tr]
            Xte, yte = Xte[idx_te], yte[idx_te]

        logger.info(f"[{day}] Window tensors: Train {Xtr.shape}, Test {Xte.shape}")

        # Compute balanced class weights with adaptive cap safely
        unique_classes, counts = np.unique(ytr, return_counts=True)
        total_samples = len(ytr)
        n_unique = len(unique_classes)
        class_weight = {}
        for c in classes:
            if c in unique_classes:
                cnt = counts[unique_classes == c][0]
                w = total_samples / (n_unique * cnt)
                class_weight[int(c)] = float(min(w, self.config.max_class_weight))
            else:
                class_weight[int(c)] = 1.0
        logger.info(f"[{day}] Class weights (capped at {self.config.max_class_weight}):")
        for c, w in class_weight.items():
            logger.info(f"   Class {c} ({labels[c]}): {w:.2f}")

        # Build model
        model = build_specialist_model(
            n_features=len(feats),
            lookback=self.config.lookback,
            n_classes=len(labels),
            config=self.config,
            name=f"ft_tcn_specialist_{day.lower()}",
        )
        optimizer = keras.optimizers.AdamW(
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            clipnorm=1.0,
        )
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                str(mdir / "best_specialist.keras"),
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

        epochs = 2 if self.config.smoke_test else self.config.epochs
        logger.info(f"[{day}] Fitting model for up to {epochs} epochs (batch={self.config.batch_size})...")
        hist = model.fit(
            Xtr,
            ytr,
            validation_split=0.15,
            epochs=epochs,
            batch_size=self.config.batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )

        logger.info(f"[{day}] Predicting test windows...")
        probs = model.predict(Xte, batch_size=2048, verbose=0)
        pred = probs.argmax(axis=1)

        # Save model and test artifacts
        model.save(final_model_path)
        joblib.dump(le, mdir / "label_encoder.joblib")
        np.save(mdir / "test_probs.npy", probs)
        np.save(mdir / "test_true.npy", yte)
        pd.DataFrame(hist.history).to_csv(mdir / "training_history.csv", index=False)

        # Compute metrics
        logger.info(f"\n===== {day} Classification Report =====")
        cls_rep_str = classification_report(yte, pred, labels=classes, target_names=labels, zero_division=0)
        logger.info("\n" + cls_rep_str)
        cls_rep_dict = classification_report(yte, pred, labels=classes, target_names=labels, zero_division=0, output_dict=True)

        cm = confusion_matrix(yte, pred, labels=classes)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(
            self.reports_dir / f"specialist_{day}_confusion_matrix.csv"
        )

        # Binary attack vs benign metrics
        benign_idx = labels.index("BENIGN") if "BENIGN" in labels else None
        attack_score = 1.0 - probs[:, benign_idx] if benign_idx is not None else probs.max(axis=1)
        ybin = (np.array(labels)[yte] != "BENIGN").astype(int)

        ap = (
            float(average_precision_score(ybin, attack_score))
            if len(np.unique(ybin)) == 2
            else np.nan
        )
        auc = (
            float(roc_auc_score(ybin, attack_score))
            if len(np.unique(ybin)) == 2
            else np.nan
        )

        # Multiclass One-vs-Rest ROC-AUC
        try:
            present_classes = np.unique(yte)
            valid_mask = np.array([c in present_classes for c in classes])
            if valid_mask.sum() > 1:
                y_binarized = label_binarize(yte, classes=classes)
                macro_auc = float(roc_auc_score(
                    y_binarized[:, valid_mask], probs[:, valid_mask], average="macro", multi_class="ovr"
                ))
            else:
                macro_auc = np.nan
        except Exception as e:
            logger.warning(f"[{day}] OvR ROC-AUC computation warning: {e}")
            macro_auc = np.nan

        metrics = {
            "day": day,
            "classes": labels,
            "n_train_windows": int(len(Xtr)),
            "n_test_windows": int(len(Xte)),
            "classification_report": cls_rep_dict,
            "attack_vs_benign_average_precision": ap,
            "attack_vs_benign_roc_auc": auc,
            "macro_multiclass_roc_auc_ovr": macro_auc,
            "split": "StratifiedShuffleSplit 70/30",
        }
        metrics_path.write_text(json.dumps(metrics, indent=2))

        # Generate plots
        self._plot_curves(day, ybin, attack_score)
        self._plot_confusion_matrix(day, cm, labels)

        logger.info(f"[{day}] Completed successfully! AP={ap:.4f}, ROC_AUC={auc:.4f}, Macro_AUC={macro_auc:.4f}")
        return metrics

    def _plot_curves(self, day: str, ybin: np.ndarray, attack_score: np.ndarray):
        """Plots and saves ROC and Precision-Recall curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        if len(np.unique(ybin)) == 2:
            fpr, tpr, _ = roc_curve(ybin, attack_score)
            pr, rc, _ = precision_recall_curve(ybin, attack_score)
            auc_val = roc_auc_score(ybin, attack_score)
            ap_val = average_precision_score(ybin, attack_score)

            axes[0].plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC (AUC = {auc_val:.4f})")
            axes[0].plot([0, 1], [0, 1], color="#94a3b8", linestyle="--")
            axes[0].set_title(f"{day} Specialist — ROC Curve", fontsize=12, fontweight="bold")
            axes[0].set_xlabel("False Positive Rate")
            axes[0].set_ylabel("True Positive Rate")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(loc="lower right")

            axes[1].plot(rc, pr, color="#16a34a", lw=2, label=f"PR (AP = {ap_val:.4f})")
            axes[1].set_title(f"{day} Specialist — Precision-Recall Curve", fontsize=12, fontweight="bold")
            axes[1].set_xlabel("Recall")
            axes[1].set_ylabel("Precision")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc="lower left")
        else:
            axes[0].text(0.5, 0.5, "Single Class", ha="center")
            axes[1].text(0.5, 0.5, "Single Class", ha="center")

        plt.tight_layout()
        out_path = self.reports_dir / f"specialist_{day}_roc_pr.png"
        plt.savefig(out_path, dpi=140)
        plt.close(fig)

    def _plot_confusion_matrix(self, day: str, cm: np.ndarray, labels: List[str]):
        """Plots and saves confusion matrix heatmap."""
        plt.figure(figsize=(max(6, len(labels) * 1.5), max(5, len(labels) * 1.2)))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
        )
        plt.title(f"{day} Specialist — Confusion Matrix Heatmap", fontsize=13, fontweight="bold")
        plt.xlabel("Predicted Label", fontweight="bold")
        plt.ylabel("Ground Truth Label", fontweight="bold")
        plt.xticks(rotation=30, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        out_path = self.reports_dir / f"specialist_{day}_confusion_matrix.png"
        plt.savefig(out_path, dpi=140)
        plt.close()

    def _ensure_plots(self, day: str, mdir: Path, metrics: Dict[str, Any]):
        """Generates plots if missing from previously trained artifacts."""
        cm_png = self.reports_dir / f"specialist_{day}_confusion_matrix.png"
        cm_csv = self.reports_dir / f"specialist_{day}_confusion_matrix.csv"
        roc_png = self.reports_dir / f"specialist_{day}_roc_pr.png"

        labels = metrics.get("classes", [])
        if cm_csv.exists() and not cm_png.exists():
            cm_df = pd.read_csv(cm_csv, index_col=0)
            self._plot_confusion_matrix(day, cm_df.to_numpy(), list(cm_df.columns))

        probs_path = mdir / "test_probs.npy"
        true_path = mdir / "test_true.npy"
        if not roc_png.exists() and probs_path.exists() and true_path.exists():
            probs = np.load(probs_path)
            yte = np.load(true_path)
            benign_idx = labels.index("BENIGN") if "BENIGN" in labels else None
            attack_score = 1.0 - probs[:, benign_idx] if benign_idx is not None else probs.max(axis=1)
            ybin = (np.array(labels)[yte] != "BENIGN").astype(int)
            self._plot_curves(day, ybin, attack_score)

    def train_all(self, days: List[str] = ("Tuesday", "Wednesday", "Thursday", "Friday"), force: bool = False) -> pd.DataFrame:
        """Trains and evaluates specialists across all designated days, saving a summary scorecard."""
        results = []
        for d in days:
            metrics = self.train_specialist(d, force=force)
            results.append({
                "day": metrics["day"],
                "classes": len(metrics["classes"]),
                "train_windows": metrics["n_train_windows"],
                "test_windows": metrics["n_test_windows"],
                "weighted_f1": metrics["classification_report"]["weighted avg"]["f1-score"],
                "weighted_recall": metrics["classification_report"]["weighted avg"]["recall"],
                "weighted_precision": metrics["classification_report"]["weighted avg"]["precision"],
                "AP": metrics["attack_vs_benign_average_precision"],
                "ROC_AUC": metrics["attack_vs_benign_roc_auc"],
                "macro_multiclass_ROC_AUC_OVR": metrics["macro_multiclass_roc_auc_ovr"],
            })

        summary = pd.DataFrame(results)
        summary_path = self.reports_dir / "specialist_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info(f"\n===== All Specialists Summary Scorecard =====\n{summary.to_string(index=False)}")
        return summary
