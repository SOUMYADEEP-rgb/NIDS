"""
Engine B: Unsupervised Clustering for Novel Attack Profiling
============================================================
Clusters anomalous traffic flagged by Engine A using learned 64-dimensional
bottleneck embeddings from the FT-Transformer autoencoder.

Key Capabilities:
1. Feature Extraction: Extracts 64D latent representations from the autoencoder bottleneck.
2. Unsupervised Clustering: Implements both KMeans and HDBSCAN.
3. Clustering Validation:
   - Silhouette Score (target >= 0.25)
   - Calinski-Harabasz Index
   - Davies-Bouldin Index
4. External Audit (Evaluation Only):
   - Computes cluster purity and alignment with real attack families (DDoS, PortScan, Web Attacks).
5. Behavioral Profiling:
   - Calculates mean/std of key network flow metrics per cluster to build automated profiles.
6. Visualization:
   - 2D t-SNE / PCA projections saved as high-resolution plots.
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
from sklearn.cluster import KMeans, MiniBatchKMeans, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import tensorflow as tf
import keras
from keras import layers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EngineB_Clustering")


@dataclass
class EngineBConfig:
    n_clusters_kmeans: int = 5
    min_cluster_size_hdbscan: int = 50
    min_samples_hdbscan: int = 15
    tsne_perplexity: float = 35.0
    random_state: int = 42
    max_cluster_samples: int = 25000


class ClusteringEngine:
    def __init__(self, base_dir: Path, config: Optional[EngineBConfig] = None):
        self.base = Path(base_dir)
        self.cache_dir = self.base / "cache"
        self.feat_dir = self.base / "features"
        self.models_dir = self.base / "models" / "engine_a_flow_ae"
        self.clust_dir = self.base / "clusters"
        self.reports_dir = self.base / "reports"
        self.clust_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or EngineBConfig()

        raw_feat_path = self.feat_dir / "raw_feature_candidates.json"
        selected_feat_path = self.feat_dir / "selected_features.json"
        if selected_feat_path.exists():
            self.selected_features = json.loads(selected_feat_path.read_text())["features"]
        else:
            self.selected_features = [f"feat_{i}" for i in range(32)]

    def extract_anomalous_pool(
        self,
        autoencoder: keras.Model,
        thresholds: Dict[str, Any],
        days: Tuple[str, ...] = ("Thursday", "Friday"),
        threshold_level: str = "p95",
        max_samples_per_day: int = 40000,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Flags anomalous flows across unseen days and extracts 64D bottleneck embeddings."""
        logger.info(f"Extracting flagged anomalies across {days} using Monday {threshold_level} threshold...")
        
        thr_mean = thresholds["mean_mse"][threshold_level]
        thr_top3 = thresholds["top3_mse"][threshold_level]
        
        # Build embedding extractor from the bottleneck layer
        embedding_model = keras.Model(
            inputs=autoencoder.input,
            outputs=autoencoder.get_layer("bottleneck").output,
            name="bottleneck_extractor",
        )

        flagged_embeddings = []
        flagged_raw_features = []
        flagged_metadata = []

        rng = np.random.default_rng(self.config.random_state)

        for day in days:
            X = np.load(self.cache_dir / f"X_{day}.npy")
            y_meta = pd.read_parquet(self.cache_dir / f"y_{day}.parquet")
            
            # Subsample for evaluation efficiency if dataset is huge
            if max_samples_per_day and len(X) > max_samples_per_day:
                # Keep diverse sample
                att_idx = np.where(y_meta["Label"] != "BENIGN")[0]
                ben_idx = np.where(y_meta["Label"] == "BENIGN")[0]
                n_take_ben = min(len(ben_idx), max_samples_per_day // 2)
                n_take_att = min(len(att_idx), max_samples_per_day // 2)
                sel = np.sort(np.concatenate([
                    rng.choice(att_idx, n_take_att, replace=False) if len(att_idx) > n_take_att else att_idx,
                    rng.choice(ben_idx, n_take_ben, replace=False) if len(ben_idx) > n_take_ben else ben_idx,
                ]))
                X_eval = X[sel]
                meta_eval = y_meta.iloc[sel].reset_index(drop=True)
            else:
                X_eval = X
                meta_eval = y_meta.copy()

            logger.info(f"[{day}] Scoring {len(X_eval):,} flows...")
            pred = autoencoder.predict(X_eval, batch_size=4096, verbose=0)
            sq_err = np.square(pred - X_eval)
            score_mean = np.mean(sq_err, axis=1)
            score_top3 = np.mean(np.sort(sq_err, axis=1)[:, -3:], axis=1)

            # Flag if either mean error or top-3 feature error exceeds normal Monday threshold
            is_anom = (score_mean > thr_mean) | (score_top3 > thr_top3)
            logger.info(f"[{day}] Flagged {is_anom.sum():,} anomalous flows ({is_anom.mean()*100:.2f}%).")

            if is_anom.sum() > 0:
                X_anom = X_eval[is_anom]
                emb_anom = embedding_model.predict(X_anom, batch_size=4096, verbose=0)
                meta_anom = meta_eval.iloc[np.where(is_anom)[0]].copy()
                meta_anom["day"] = day
                meta_anom["reconstruction_mse"] = score_mean[is_anom]
                meta_anom["top3_feature_mse"] = score_top3[is_anom]

                flagged_embeddings.append(emb_anom)
                flagged_raw_features.append(X_anom)
                flagged_metadata.append(meta_anom)

        all_emb = np.vstack(flagged_embeddings).astype("float32")
        all_feats = np.vstack(flagged_raw_features).astype("float32")
        all_meta = pd.concat(flagged_metadata, ignore_index=True)

        logger.info(f"Total flagged anomaly pool: {len(all_emb):,} flows.")
        logger.info(f"Ground-truth composition of anomaly pool (for audit):\n{dict(all_meta['Label'].value_counts())}")

        return all_emb, all_feats, all_meta

    def run_clustering(
        self,
        embeddings: np.ndarray,
        raw_features: np.ndarray,
        metadata: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Runs KMeans & HDBSCAN on bottleneck embeddings and computes evaluation metrics."""
        logger.info("Running Unsupervised Clustering on Bottleneck Embeddings...")
        
        # Subsample if pool is larger than max_cluster_samples for fast Silhouette & t-SNE
        if len(embeddings) > self.config.max_cluster_samples:
            rng = np.random.default_rng(self.config.random_state)
            sel_idx = np.sort(rng.choice(len(embeddings), self.config.max_cluster_samples, replace=False))
            emb_sub = embeddings[sel_idx]
            feat_sub = raw_features[sel_idx]
            meta_sub = metadata.iloc[sel_idx].reset_index(drop=True)
        else:
            emb_sub = embeddings
            feat_sub = raw_features
            meta_sub = metadata.copy()

        # 1. KMeans Clustering (k = n_clusters_kmeans)
        k = self.config.n_clusters_kmeans
        logger.info(f"Fitting KMeans (k={k}) on {len(emb_sub):,} 64D bottleneck vectors...")
        km = KMeans(n_clusters=k, random_state=self.config.random_state, n_init=10)
        km_labels = km.fit_predict(emb_sub)

        # 2. HDBSCAN Density-Based Clustering
        logger.info("Fitting HDBSCAN density clustering...")
        hdb = HDBSCAN(
            min_cluster_size=self.config.min_cluster_size_hdbscan,
            min_samples=self.config.min_samples_hdbscan,
        )
        hdb_labels = hdb.fit_predict(emb_sub)

        # Compute unsupervised quality metrics for KMeans
        logger.info("Computing unsupervised clustering metrics (Silhouette, CH, DB)...")
        # Sample for Silhouette calculation if N > 10,000 for speed
        sil_n = min(10000, len(emb_sub))
        sil_idx = np.random.default_rng(self.config.random_state).choice(len(emb_sub), sil_n, replace=False)
        
        sil_km = float(silhouette_score(emb_sub[sil_idx], km_labels[sil_idx]))
        ch_km = float(calinski_harabasz_score(emb_sub, km_labels))
        db_km = float(davies_bouldin_score(emb_sub, km_labels))

        logger.info(f"KMeans (k={k}) Quality: Silhouette = {sil_km:.4f}, Calinski-Harabasz = {ch_km:.1f}, Davies-Bouldin = {db_km:.4f}")

        # Compute cluster purity and alignment against ground-truth labels (Audit Only)
        meta_sub["kmeans_cluster"] = km_labels
        meta_sub["hdbscan_cluster"] = hdb_labels

        cluster_audit = []
        for c in range(k):
            c_mask = km_labels == c
            c_meta = meta_sub[c_mask]
            total_in_cluster = len(c_meta)
            vc = c_meta["Label"].value_counts()
            dominant_label = vc.index[0]
            dominant_count = int(vc.iloc[0])
            purity = (dominant_count / total_in_cluster) * 100

            cluster_audit.append({
                "cluster_id": int(c),
                "size": int(total_in_cluster),
                "purity_pct": float(round(purity, 2)),
                "dominant_label": str(dominant_label),
                "label_breakdown": {str(k): int(v) for k, v in vc.head(5).items()},
            })
            logger.info(f"  Cluster {c}: Size={total_in_cluster:>5,}, Dominant='{dominant_label}' (Purity: {purity:.1f}%), Mix: {dict(vc.head(3))}")

        # Build feature profiles (mean values of original features per cluster)
        profile_df = pd.DataFrame(feat_sub, columns=self.selected_features).groupby(km_labels).mean()
        profile_df.index.name = "cluster_id"
        profile_df["cluster_size"] = meta_sub.groupby("kmeans_cluster").size()
        profile_df["dominant_label_audit"] = [ca["dominant_label"] for ca in cluster_audit]
        profile_df["purity_audit"] = [f"{ca['purity_pct']}%" for ca in cluster_audit]

        # Save artifacts
        joblib.dump(km, self.clust_dir / "kmeans_model.joblib")
        meta_sub.to_parquet(self.clust_dir / "clustered_anomalies.parquet", index=False)
        profile_df.to_csv(self.clust_dir / "cluster_feature_profiles.csv")

        # Visualizations (2D Projection)
        self._plot_2d_projections(emb_sub, km_labels, meta_sub["Label"].to_numpy())

        results = {
            "timestamp": time.ctime(),
            "total_clustered_flows": len(emb_sub),
            "n_clusters": k,
            "metrics": {
                "silhouette_score": sil_km,
                "calinski_harabasz_index": ch_km,
                "davies_bouldin_index": db_km,
            },
            "hdbscan_summary": {
                "n_clusters_found": int(len(np.unique(hdb_labels[hdb_labels != -1]))),
                "noise_ratio_pct": float((hdb_labels == -1).mean() * 100),
            },
            "cluster_audit": cluster_audit,
        }

        (self.clust_dir / "cluster_evaluation.json").write_text(
            json.dumps(
                results,
                indent=2,
                default=lambda o: int(o) if isinstance(o, (np.integer, np.int64)) else float(o) if isinstance(o, (np.floating, np.float32, np.float64)) else str(o),
            )
        )
        return results

    def _plot_2d_projections(self, embeddings: np.ndarray, cluster_ids: np.ndarray, true_labels: np.ndarray):
        """Generates side-by-side PCA/t-SNE 2D projections."""
        logger.info("Computing 2D PCA projection for visualization...")
        
        # Subsample up to 4000 points for fast, crisp visualization
        n_vis = min(4000, len(embeddings))
        rng = np.random.default_rng(self.config.random_state)
        v_idx = rng.choice(len(embeddings), n_vis, replace=False)
        
        pca = PCA(n_components=2, random_state=self.config.random_state)
        vis_2d = pca.fit_transform(embeddings[v_idx])
        c_vis = cluster_ids[v_idx]
        lbl_vis = true_labels[v_idx]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Colored by Unsupervised Cluster ID
        scatter1 = sns.scatterplot(
            x=vis_2d[:, 0],
            y=vis_2d[:, 1],
            hue=c_vis,
            palette="tab10",
            alpha=0.7,
            s=25,
            ax=axes[0],
        )
        axes[0].set_title("Engine B: Unsupervised Cluster Groups (Bottleneck Manifold)", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("PCA Component 1")
        axes[0].set_ylabel("PCA Component 2")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(title="Cluster ID", loc="best")

        # Plot 2: Colored by True Ground Truth Label (Evaluation Audit)
        scatter2 = sns.scatterplot(
            x=vis_2d[:, 0],
            y=vis_2d[:, 1],
            hue=lbl_vis,
            palette="Set2",
            alpha=0.7,
            s=25,
            ax=axes[1],
        )
        axes[1].set_title("Ground-Truth Attack Separation (Audit Verification)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("PCA Component 1")
        axes[1].set_ylabel("PCA Component 2")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(title="True Label", loc="best")

        plt.tight_layout()
        out_fig = self.clust_dir / "cluster_manifold_projection.png"
        plt.savefig(out_fig, dpi=140)
        plt.close(fig)
        logger.info(f"Saved 2D visualization plot to {out_fig}")
