import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


def calculate_metrics(y_true, y_pred, save_folder, model_i, average):
    """
    Calculates classification metrics (accuracy, F1-score, precision, and recall)
    for a given model's predictions and saves the results as a CSV file.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.
        save_folder (str): Path to the folder where the metrics CSV file will be saved.
        model_i (int): Model identifier (used in the filename).
        average (str): Averaging method for multi-class classification metrics
                       (e.g., "weighted", "macro", "micro").

    Saves:
        A CSV file named `model{model_i}_metrics_{average}.csv` containing the computed metrics.
    """
    acc_score = accuracy_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred, average=average) * 100
    precision = precision_score(y_true, y_pred, average=average) * 100
    recall = recall_score(y_true, y_pred, average=average) * 100
    df_dict = {
        "model": model_i,
        "accuracy": [round(acc_score, 2)],
        f"f1-score_{average}": [round(f1, 2)],
        f"precision_{average}": [round(precision, 2)],
        f"recall_{average}": [round(recall, 2)],
    }
    pd.DataFrame(df_dict).to_csv(
        f"{save_folder}/model{model_i}_metrics_{average}.csv", index=False
    )
