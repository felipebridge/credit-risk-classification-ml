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
    vals = np.unique(y)
    return len(vals) == 2


def evaluate_and_save_metrics(model: Any, X_test, y_test, output_path) -> None:
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    avg = "binary" if _is_binary(y_test) else "weighted"

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average=avg,
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    auc_text = "ROC-AUC: N/A"
    if _is_binary(y_test):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            auc_text = f"ROC-AUC: {auc:.4f}"
        except Exception:
            pass

    lines = []
    lines.append("=== Credit Approval Model Evaluation ===")
    lines.append(f"Accuracy: {acc:.4f}")
    lines.append(f"Precision ({avg}): {prec:.4f}")
    lines.append(f"Recall ({avg}): {rec:.4f}")
    lines.append(f"F1 ({avg}): {f1:.4f}")
    lines.append(auc_text)
    lines.append("")
    lines.append("Confusion Matrix:")
    lines.append(str(cm))
    lines.append("")
    lines.append("Classification Report:")
    lines.append(report)

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")