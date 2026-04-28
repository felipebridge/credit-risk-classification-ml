import argparse

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split

from src.config import (
    CASE_STUDY_1_PATH,
    CASE_STUDY_2_PATH,
    ID_COL,
    TARGET_COL,
    MISSING_VALUE_SENTINEL,
    RANDOM_SEED,
    TEST_SIZE,
    MODELS_DIR,
    REPORTS_DIR,
    MODEL_PATH,
    CREDIT_SCORE_COL,
    METRICS_WITH_SCORE_PATH,
    METRICS_NO_SCORE_PATH,
    FEATURE_IMPORTANCE_WITH_SCORE_PATH,
    FEATURE_IMPORTANCE_NO_SCORE_PATH,
)
from src.pipeline import build_model_pipeline
from src.evaluate import evaluate_and_save_metrics


def load_and_merge() -> pd.DataFrame:
    df1 = pd.read_excel(CASE_STUDY_1_PATH)
    df2 = pd.read_excel(CASE_STUDY_2_PATH)
    return df2.merge(df1, on=ID_COL, how="inner")


def get_feature_names(pipe, X_train: pd.DataFrame) -> list[str]:
    preprocess = pipe.named_steps["preprocess"]
    num_cols = preprocess.transformers_[0][2]
    cat_cols = preprocess.transformers_[1][2]

    names = [str(c) for c in num_cols]

    ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
    names.extend(str(n) for n in ohe.get_feature_names_out(cat_cols))

    return names


def export_feature_importance(pipe, X_train: pd.DataFrame, output_path) -> None:
    model = pipe.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return

    names = get_feature_names(pipe, X_train)
    importances = model.feature_importances_
    n = min(len(names), len(importances))

    (
        pd.DataFrame({"feature": names[:n], "importance": importances[:n]})
        .sort_values("importance", ascending=False)
        .to_csv(output_path, index=False, encoding="utf-8")
    )


def main(drop_credit_score: bool = False) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_merge()
    df = df.replace(MISSING_VALUE_SENTINEL, np.nan)

    if TARGET_COL not in df.columns:
        raise ValueError(f"No se encontró la columna target '{TARGET_COL}'.")

    df = df[df[TARGET_COL].notna()].copy()

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    if ID_COL in X.columns:
        X = X.drop(columns=[ID_COL])

    if drop_credit_score and CREDIT_SCORE_COL in X.columns:
        X = X.drop(columns=[CREDIT_SCORE_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )

    pipe = build_model_pipeline(X_train, RANDOM_SEED)
    pipe.fit(X_train, y_train)

    metrics_path = METRICS_NO_SCORE_PATH if drop_credit_score else METRICS_WITH_SCORE_PATH
    fi_path = FEATURE_IMPORTANCE_NO_SCORE_PATH if drop_credit_score else FEATURE_IMPORTANCE_WITH_SCORE_PATH

    dump(pipe, MODEL_PATH)
    evaluate_and_save_metrics(pipe, X_test, y_test, metrics_path)
    export_feature_importance(pipe, X_train, fi_path)

    escenario = "SIN Credit_Score (ablation study)" if drop_credit_score else "CON Credit_Score"
    print(f"Escenario:                    {escenario}")
    print(f"Modelo guardado en:           {MODEL_PATH}")
    print(f"Métricas guardadas en:        {metrics_path}")
    print(f"Importancia de variables en:  {fi_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrena el modelo de clasificación de riesgo crediticio."
    )
    parser.add_argument(
        "--no-credit-score",
        action="store_true",
        help="Excluir Credit_Score del entrenamiento (ablation study).",
    )
    args = parser.parse_args()
    main(drop_credit_score=args.no_credit_score)
