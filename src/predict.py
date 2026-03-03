import numpy as np
import pandas as pd
from joblib import load

from src.config import (
    MODEL_PATH,
    ID_COL,
    TARGET_COL,
    MISSING_VALUE_SENTINEL,
)


def predict_from_excel(input_path: str, output_path: str) -> None:
    model = load(MODEL_PATH)

    df = pd.read_excel(input_path)
    df = df.replace(MISSING_VALUE_SENTINEL, np.nan)

    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    ids = None
    if ID_COL in df.columns:
        ids = df[ID_COL]
        X = df.drop(columns=[ID_COL])
    else:
        X = df

    preds = model.predict(X)

    out = pd.DataFrame({"prediction_Approved_Flag": preds})
    if ids is not None:
        out.insert(0, ID_COL, ids)

    out.to_csv(output_path, index=False)


if __name__ == "__main__":
    predict_from_excel("data/raw/case_study2.xlsx", "reports/predictions.csv")
    print("Predicciones guardadas en reports/predictions.csv")