#!/usr/bin/env python3
"""Minimal project launcher for the hybrid NIDS pipeline.

This is the single entrypoint for the repo. It keeps the working Python engine
scripts as the source of truth and exposes a clean CLI around the production
flow already implemented in:

- production_hybrid_nids/engine_a_flow_autoencoder.py
- production_hybrid_nids/engine_b_clustering.py
- production_hybrid_nids/engine_c_specialists.py
- production_hybrid_nids/section5_temporal_generalization.py

Examples:
    python run_hybrid_nids.py --mode engine_a
    python run_hybrid_nids.py --mode engine_b
    python run_hybrid_nids.py --mode engine_c
    python run_hybrid_nids.py --mode section5
    python run_hybrid_nids.py --mode all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT / "production_hybrid_nids"


def ensure_project_ready() -> None:
    required = [
        ROOT / "production_hybrid_nids",
        ROOT / "production_hybrid_nids" / "cache",
        ROOT / "production_hybrid_nids" / "features",
        ROOT / "production_hybrid_nids" / "models",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing expected project paths. Please verify the project structure before running the pipeline.\n"
            + "\n".join(f" - {item}" for item in missing)
        )


def run_engine_a(force: bool = False):
    from production_hybrid_nids.engine_a_flow_autoencoder import EngineAFlowTrainer

    trainer = EngineAFlowTrainer(PROJECT_DIR)
    return trainer.train_or_load(force=force)


def run_engine_b(force: bool = False):
    from production_hybrid_nids.engine_a_flow_autoencoder import EngineAFlowTrainer
    from production_hybrid_nids.engine_b_clustering import ClusteringEngine

    model, thresholds = run_engine_a(force=force)
    engine_b = ClusteringEngine(PROJECT_DIR)
    embeddings, raw_features, metadata = engine_b.extract_anomalous_pool(
        model,
        thresholds,
        days=("Thursday", "Friday"),
        threshold_level="p95",
    )
    return engine_b.run_clustering(embeddings, raw_features, metadata)


def run_engine_c(day: str | None = None, force: bool = False):
    from production_hybrid_nids.engine_c_specialists import SpecialistEngine

    engine = SpecialistEngine(PROJECT_DIR)
    if day:
        return engine.train_specialist(day, force=force)
    return engine.train_all(force=force)


def run_section5(force: bool = False):
    from production_hybrid_nids.section5_temporal_generalization import TemporalGeneralizationEngine

    engine = TemporalGeneralizationEngine(PROJECT_DIR)
    return engine.train_all()


def run_full_pipeline(force: bool = False):
    print("\n[1/4] Running Engine A: anomaly detection training/evaluation")
    model, thresholds = run_engine_a(force=force)

    print("\n[2/4] Running Engine B: anomaly clustering")
    engine_b = __import__("production_hybrid_nids.engine_b_clustering", fromlist=["ClusteringEngine"]).ClusteringEngine(PROJECT_DIR)
    embeddings, raw_features, metadata = engine_b.extract_anomalous_pool(
        model,
        thresholds,
        days=("Thursday", "Friday"),
        threshold_level="p95",
    )
    engine_b.run_clustering(embeddings, raw_features, metadata)

    print("\n[3/4] Running Engine C: day-specific specialists")
    run_engine_c(force=force)

    print("\n[4/4] Running Section 5: temporal generalization")
    run_section5(force=force)

    print("\nHybrid NIDS pipeline completed successfully.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the hybrid NIDS pipeline using the canonical Python engine scripts."
    )
    parser.add_argument(
        "--mode",
        choices=["engine_a", "engine_b", "engine_c", "section5", "all"],
        default="all",
        help="Which pipeline stage to run. Defaults to the full sequence.",
    )
    parser.add_argument(
        "--day",
        type=str,
        default=None,
        help="Optional single-day target for Engine C (for example: Tuesday, Wednesday, Thursday, Friday).",
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=str(PROJECT_DIR),
        help="Path to the production_hybrid_nids directory. Defaults to the repo's canonical project folder.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining of available models even if prior artifacts exist.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    if project_dir.exists():
        global PROJECT_DIR
        PROJECT_DIR = project_dir

    try:
        ensure_project_ready()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        if args.mode == "engine_a":
            run_engine_a(force=args.force)
        elif args.mode == "engine_b":
            run_engine_b(force=args.force)
        elif args.mode == "engine_c":
            run_engine_c(day=args.day, force=args.force)
        elif args.mode == "section5":
            run_section5(force=args.force)
        else:
            run_full_pipeline(force=args.force)
    except Exception as exc:  # pragma: no cover - CLI safety wrapper
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
