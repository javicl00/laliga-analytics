"""Analisis de importancia de features del modelo guardado.

Imprime tabla y grafico ASCII en consola. No requiere matplotlib.
Guarda PNG solo si matplotlib esta disponible en el entorno.

Uso:
    docker compose run --rm etl python -m src.training.feature_importance
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_BAR_WIDTH = 40  # caracteres del grafico ASCII


def _get_lgbm(model) -> LGBMClassifier | None:
    if isinstance(model, Pipeline):
        for _, step in model.steps:
            if isinstance(step, LGBMClassifier):
                return step
        return None
    return model if isinstance(model, LGBMClassifier) else None


def _ascii_bar(value: float, max_value: float) -> str:
    filled = int(round(value / max_value * _BAR_WIDTH))
    return "█" * filled + "░" * (_BAR_WIDTH - filled)


def _print_table(df: pd.DataFrame, value_col: str, label: str) -> None:
    max_val = df[value_col].max()
    feat_w  = max(len(f) for f in df["feature"]) + 2
    print(f"\n{'#':>3}  {'Feature':<{feat_w}}  {label:>10}  {'%':>6}  Bar")
    print("-" * (feat_w + _BAR_WIDTH + 28))
    for i, row in df.iterrows():
        bar = _ascii_bar(row[value_col], max_val)
        print(f"{i:>3}. {row['feature']:<{feat_w}}  {row[value_col]:>10.2f}  {row[value_col] / max_val * 100:>5.1f}%  {bar}")


def _try_save_png(df: pd.DataFrame, value_col: str, label: str, output_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(df)))
        fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.4)))
        ax.barh(df["feature"][::-1], df[value_col][::-1], color=colors[::-1])
        ax.set_xlabel(label)
        ax.set_title(f"Feature Importance — {label}")
        plt.tight_layout()
        out_path = Path(output_dir) / "feature_importance.png"
        plt.savefig(out_path, dpi=150)
        logger.info("PNG guardado en %s", out_path)
        plt.close()
    except ModuleNotFoundError:
        logger.debug("matplotlib no disponible, omitiendo PNG")


def run(model_path: str = "models/lgbm_v1.pkl", output_dir: str = "models") -> pd.DataFrame:
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model        = bundle["model"]
    feature_cols = bundle["feature_cols"]

    lgbm = _get_lgbm(model)

    if lgbm is None:
        # LogisticRegression: coeficiente absoluto medio entre clases
        from sklearn.linear_model import LogisticRegression
        lr = model.named_steps["lr"] if isinstance(model, Pipeline) else model
        if not isinstance(lr, LogisticRegression):
            raise ValueError(f"Tipo de modelo no soportado: {type(model)}")
        coef_mean = np.abs(lr.coef_).mean(axis=0)
        df = pd.DataFrame({
            "feature":    feature_cols,
            "coef_abs":   coef_mean,
        }).sort_values("coef_abs", ascending=False).reset_index(drop=True)
        df.index += 1
        _print_table(df, "coef_abs", "Coef abs medio")
        _try_save_png(df, "coef_abs", "Coef abs medio", output_dir)
        return df

    # LightGBM
    gain   = lgbm.booster_.feature_importance(importance_type="gain")
    splits = lgbm.booster_.feature_importance(importance_type="split")

    df = pd.DataFrame({
        "feature": feature_cols,
        "gain":    gain,
        "splits":  splits,
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    df.index += 1

    print("\n" + "=" * 80)
    print(" FEATURE IMPORTANCE — GAIN (valor predictivo total, mayor = mejor)")
    print("=" * 80)
    _print_table(df, "gain", "Gain")

    print("\n" + "=" * 80)
    print(" FEATURE IMPORTANCE — SPLITS (frecuencia de uso en arboles)")
    print("=" * 80)
    df_splits = df.sort_values("splits", ascending=False).reset_index(drop=True)
    df_splits.index += 1
    _print_table(df_splits, "splits", "Splits")

    _try_save_png(df, "gain", "Gain", output_dir)
    return df


if __name__ == "__main__":
    run()
