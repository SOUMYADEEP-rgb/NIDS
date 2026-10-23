#!/usr/bin/env python3
"""Small, isolated random-search extension for Engine A and Engine C.

This script deliberately does not modify production models, reports, or feature
pipelines. Trial outputs are written below ``tuning_results/`` only.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

from production_hybrid_nids.engine_a_flow_autoencoder import (
    FlowAEConfig,
    build_flow_ft_autoencoder,
)
from production_hybrid_nids.engine_c_specialists import (
    EngineCConfig,
    SpecialistEngine,
    build_specialist_model,
    make_windows,
    slog,
)


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "production_hybrid_nids"


def choose(rng: random.Random, values: list[Any]) -> Any:
    return rng.choice(values)


def engine_a_trial(base: Path, config: FlowAEConfig, smoke: bool) -> dict[str, Any]:
    cache = base / "cache"
    X = np.load(cache / "X_Monday.npy").astype("float32")
    rng = np.random.default_rng(config.seed)
    if smoke:
        X = X[: min(20_000, len(X))]
    order = rng.permutation(len(X))
    n_val = max(1, int(0.20 * len(X)))
    va, tr = order[:n_val], order[n_val:]

    model = build_flow_ft_autoencoder(config)
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            clipnorm=1.0,
        ),
        loss="mse",
    )
    epochs = 2 if smoke else config.epochs
    history = model.fit(
        X[tr], X[tr], validation_data=(X[va], X[va]),
        epochs=epochs, batch_size=config.batch_size, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.patience,
            restore_best_weights=True,
        )],
    )
    val_loss = float(min(history.history["val_loss"]))
    return {"val_loss": val_loss, "objective": val_loss}


def engine_c_trial(base: Path, day: str, config: EngineCConfig, smoke: bool) -> dict[str, Any]:
    """Tune the classifier using the existing frozen train-split pipeline.

    The production test split is never used for trial selection. The Keras
    validation split is taken from the training windows only.
    """
    engine = SpecialistEngine(base, config=config)
    df = engine.load_day(day)
    tr_idx, te_idx = engine.stratified_day_split(df)
    fdir = base / "features" / f"specialist_{day}"
    selected = json.loads((fdir / "selected_features.json").read_text())["features"]
    pipeline = joblib.load(fdir / "pipeline.joblib")
    med, scaler = pipeline["medians"], pipeline["scaler"]
    X = np.clip(scaler.transform(slog(df[selected].fillna(med).to_numpy("float32"))), -5.0, 5.0).astype("float32")

    le = LabelEncoder().fit(df["Label"].iloc[tr_idx])
    tr_idx = tr_idx[np.isin(df["Label"].iloc[tr_idx], le.classes_)]
    if smoke:
        # Limit rows before constructing overlapping windows; otherwise the
        # full day can consume hundreds of MB even in a smoke test.
        tr_idx = tr_idx[: min(30_000, len(tr_idx))]
    y = le.transform(df["Label"].iloc[tr_idx])
    Xtr, ytr, _ = make_windows(X[tr_idx], y, config.lookback)
    if smoke:
        n = min(config.smoke_samples, len(Xtr))
        Xtr, ytr = Xtr[:n], ytr[:n]

    classes, counts = np.unique(ytr, return_counts=True)
    weights = {
        int(c): float(min(len(ytr) / (len(classes) * count), config.max_class_weight))
        for c, count in zip(classes, counts)
    }
    model = build_specialist_model(
        n_features=Xtr.shape[-1], lookback=config.lookback,
        n_classes=len(le.classes_), config=config,
        name=f"tuning_{day.lower()}",
    )
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            clipnorm=1.0,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    epochs = 2 if smoke else config.epochs
    history = model.fit(
        Xtr, ytr, validation_split=0.15, epochs=epochs,
        batch_size=config.batch_size, class_weight=weights, verbose=1,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.patience,
            restore_best_weights=True,
        )],
    )
    val_count = max(1, int(0.15 * len(Xtr)))
    probs = model.predict(Xtr[-val_count:], verbose=0)
    pred = probs.argmax(axis=1)
    y_val = ytr[-len(pred):]
    macro_f1 = float(f1_score(y_val, pred, average="macro", zero_division=0))
    return {
        "val_macro_f1": macro_f1,
        "val_loss": float(min(history.history["val_loss"])),
        "objective": macro_f1,
    }


def run_engine_a(trials: int, seed: int, smoke: bool, out: Path, epochs: int | None = None) -> None:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    target_epochs = epochs if epochs is not None else (4 if smoke else 12)
    for trial in range(1, trials + 1):
        cfg = FlowAEConfig(
            d_token=choose(rng, [32, 48, 64]),
            n_heads=choose(rng, [2, 4, 8]),
            n_blocks=choose(rng, [1, 2, 3]),
            bottleneck_dim=choose(rng, [32, 64, 96]),
            dropout=choose(rng, [0.05, 0.1, 0.2]),
            learning_rate=choose(rng, [3e-4, 7e-4, 1e-3]),
            batch_size=choose(rng, [512, 1024]),
            epochs=target_epochs,
            seed=seed,
        )
        started = time.time()
        print(f"\n--- Starting Engine A trial {trial}/{trials} (epochs={target_epochs}) ---", flush=True)
        try:
            result = engine_a_trial(BASE, cfg, smoke)
            row = {"trial": trial, "status": "ok", **vars(cfg), **result}
        except Exception as exc:  # keep later trials running
            row = {"trial": trial, "status": "error", "error": repr(exc), **vars(cfg)}
        row["seconds"] = round(time.time() - started, 2)
        rows.append(row)
        print(f"Engine A trial {trial}/{trials}: {row['status']} objective={row.get('objective')} (took {row['seconds']}s)", flush=True)
        df = pd.DataFrame(rows)
        df.to_csv(out / "engine_a_trials.csv", index=False)
        good = df[df.status == "ok"].sort_values("objective")
        if not good.empty:
            (out / "engine_a_best_params.json").write_text(json.dumps(good.iloc[0].to_dict(), indent=2, default=str))


def run_engine_c(day: str, trials: int, seed: int, smoke: bool, out: Path, epochs: int | None = None) -> None:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    target_epochs = epochs if epochs is not None else (4 if smoke else 20)
    for trial in range(1, trials + 1):
        cfg = EngineCConfig(
            lookback=choose(rng, [1, 5, 10]),
            batch_size=choose(rng, [256, 512, 1024]),
            epochs=target_epochs,
            patience=3 if smoke else min(5, target_epochs),
            d_token=choose(rng, [32, 48, 64]),
            n_heads=choose(rng, [2, 4, 8]),
            n_ft_blocks=choose(rng, [1, 2, 3]),
            tcn_filters=choose(rng, [64, 96, 128]),
            tcn_kernel_size=choose(rng, [3, 5, 7]),
            dropout=choose(rng, [0.05, 0.1, 0.2, 0.3]),
            learning_rate=choose(rng, [2e-4, 5e-4, 1e-3]),
            weight_decay=choose(rng, [1e-5, 1e-4, 1e-3]),
            seed=seed,
            smoke_test=smoke,
        )
        started = time.time()
        print(f"\n--- Starting Engine C ({day}) trial {trial}/{trials} (epochs={target_epochs}) ---", flush=True)
        try:
            result = engine_c_trial(BASE, day, cfg, smoke)
            row = {"trial": trial, "day": day, "status": "ok", **vars(cfg), **result}
        except Exception as exc:
            row = {"trial": trial, "day": day, "status": "error", "error": repr(exc), **vars(cfg)}
        row["seconds"] = round(time.time() - started, 2)
        rows.append(row)
        print(f"Engine C ({day}) trial {trial}/{trials}: {row['status']} objective={row.get('objective')} (took {row['seconds']}s)", flush=True)
        # Save incrementally after each trial so progress is never lost
        df = pd.DataFrame(rows)
        df.to_csv(out / f"engine_c_{day.lower()}_trials.csv", index=False)
        good = df[df.status == "ok"].sort_values("objective", ascending=False)
        if not good.empty:
            (out / f"engine_c_{day.lower()}_best_params.json").write_text(json.dumps(good.iloc[0].to_dict(), indent=2, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated random-search tuning for Engine A and Engine C.")
    parser.add_argument("--engine", choices=["a", "c", "both"], default="both")
    parser.add_argument("--day", default="Tuesday", help="Engine C day used for tuning.")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=None, help="Explicit maximum epochs per trial (e.g. 2, 3, 5).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="Use small data and two epochs per trial.")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    out = ROOT / "tuning_results"
    out.mkdir(exist_ok=True)
    if args.engine in {"a", "both"}:
        run_engine_a(args.trials, args.seed, args.smoke_test, out, epochs=args.epochs)
    if args.engine in {"c", "both"}:
        run_engine_c(args.day, args.trials, args.seed, args.smoke_test, out, epochs=args.epochs)
    print(f"Tuning outputs written to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
