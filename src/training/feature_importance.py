"""Analisis de importancia de features del modelo guardado.

Carga models/lgbm_v1.pkl y genera:
  - Tabla ordenada por importancia (gain y split count)
  - PNG en models/feature_importance.png

Uso:
    docker compose run --rm etl python -m src.training.feature_importance
"""
from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_lgbm(model) -> LGBMClassifier | None:
    """Extrae LGBMClassifier desde un Pipeline o directamente."""
    if isinstance(model, Pipeline):
        for _, step in model.steps:
            if isinstance(step, LGBMClassifier):
                return step
        return None
    if isinstance(model, LGBMClassifier):
        return model
    return None


def run(model_path: str = "models/lgbm_v1.pkl", output_dir: str = "models") -> pd.DataFrame:
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model       = bundle["model"]
    feature_cols = bundle["feature_cols"]

    lgbm = _get_lgbm(model)
    if lgbm is None:
        logger.warning(
            "El modelo guardado (%s) no es LGBMClassifier ni Pipeline con LGBM. "
            "No se pueden obtener importancias nativas.",
            type(model).__name__,
        )
        # Para LogisticRegression: usar coeficientes medios entre clases
        from sklearn.linear_model import LogisticRegression
        lr = model.named_steps["lr"] if isinstance(model, Pipeline) else model
        if isinstance(lr, LogisticRegression):
            coef_mean = np.abs(lr.coef_).mean(axis=0)
            df = pd.DataFrame({
                "feature":    feature_cols,
                "importance": coef_mean,
                "type":       "coef_abs_mean",
            }).sort_values("importance", ascending=False).reset_index(drop=True)
            df.index += 1
            print("\n=== Importancia por coeficiente absoluto medio (Logistic) ===")
            print(df.to_string())
            _plot(df, "importance", "Coef abs medio", output_dir)
            return df
        raise ValueError(f"Tipo de modelo no soportado: {type(model)}")

    # LightGBM: gain (valor predictivo) y split (frecuencia de uso)
    gain   = lgbm.booster_.feature_importance(importance_type="gain")
    splits = lgbm.booster_.feature_importance(importance_type="split")

    df = pd.DataFrame({
        "feature":    feature_cols,
        "gain":       gain,
        "gain_pct":   gain / gain.sum() * 100,
        "splits":     splits,
        "splits_pct": splits / splits.sum() * 100,
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    df.index += 1

    print("\n=== Feature Importance (LightGBM) ===")
    print(df.to_string(float_format="{:.2f}".format))
    _plot(df, "gain", "Gain", output_dir)
    return df


def _plot(df: pd.DataFrame, value_col: str, label: str, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(df)))
    bars = ax.barh(df["feature"][::-1], df[value_col][::-1], color=colors[::-1])
    ax.set_xlabel(label)
    ax.set_title(f"Feature Importance — {label}")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    plt.tight_layout()
    out_path = Path(output_dir) / "feature_importance.png"
    plt.savefig(out_path, dpi=150)
    logger.info("Guardado en %s", out_path)
    plt.close()


if __name__ == "__main__":
    run()
