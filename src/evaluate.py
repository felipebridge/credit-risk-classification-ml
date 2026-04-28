from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def _is_binary(y) -> bool:
    return len(np.unique(y)) == 2


def evaluate_and_save_metrics(model: Any, X_test, y_test, output_path) -> None:
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    avg = "binary" if _is_binary(y_test) else "weighted"

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average=avg, zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    auc_text = "ROC-AUC: N/A"
    try:
        y_proba = model.predict_proba(X_test)
        if _is_binary(y_test):
            auc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        auc_text = f"ROC-AUC (OvR ponderado): {auc:.4f}"
    except Exception:
        pass

    lines = [
        "=== Evaluación del Modelo de Aprobación Crediticia ===",
        f"Accuracy:              {acc:.4f}",
        f"Precision ({avg}):  {prec:.4f}",
        f"Recall ({avg}):     {rec:.4f}",
        f"F1 ({avg}):         {f1:.4f}",
        auc_text,
        "",
        "Matriz de Confusión:",
        str(cm),
        "",
        "Reporte de Clasificación:",
        report,
    ]

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
