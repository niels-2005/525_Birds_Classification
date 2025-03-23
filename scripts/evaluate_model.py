import os

from calculate_metrics import calculate_metrics
from get_data import get_test_dataset
from get_labels import get_actual_labels, get_predicted_labels
from load_model import load_model

EVALUATION_CONFIG = {
    "metrics_save_folder": "../model_evaluation",
    "metrics_average": "weighted",
    "model_paths": [
        "../trained_model_1/model.keras",
        "../trained_model_2/model.keras",
        "../trained_model_3/model.keras",
        "../trained_model_4/model.keras",
    ],
}


def check_if_folder_exist():
    """
    Checks if the metrics save folder exists. If not, creates the folder.

    Uses the path specified in EVALUATION_CONFIG["metrics_save_folder"].
    """
    if not os.path.exists(EVALUATION_CONFIG["metrics_save_folder"]):
        os.makedirs(EVALUATION_CONFIG["metrics_save_folder"])


def evaluate_model():
    """
    Evaluates multiple models by computing classification metrics on a test dataset.

    The function:
    - Loads the test dataset and ground truth labels.
    - Ensures the metrics save folder exists.
    - Iterates through the models specified in EVALUATION_CONFIG["model_paths"].
    - Loads each model, generates predictions, and calculates evaluation metrics.
    - Saves the computed metrics to a CSV file.

    Uses configurations from EVALUATION_CONFIG.
    """
    test_dataset = get_test_dataset()
    y_true = get_actual_labels(test_dataset)
    check_if_folder_exist()

    for i, path in enumerate(EVALUATION_CONFIG["model_paths"]):
        model = load_model(path)
        y_pred = get_predicted_labels(model, test_dataset)
        calculate_metrics(
            y_true,
            y_pred,
            save_folder=EVALUATION_CONFIG["metrics_save_folder"],
            model_i=i + 1,
            average=EVALUATION_CONFIG["metrics_average"],
        )


if __name__ == "__main__":
    evaluate_model()
