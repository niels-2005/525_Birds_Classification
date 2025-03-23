import argparse

import tensorflow as tf
from get_ds_class_names import get_ds_class_names
from get_image import get_image
from get_prediction_class_name import get_prediction_class_name
from load_model import load_model

PREDICT_SAMPLE_CONFIG = {
    "model_path": "../trained_model/model.keras",
    "image_size": [224, 224],
    "image_channels": 3,
}


def predict_sample(path):
    """
    Loads an image, runs a prediction, and prints the predicted class.

    The function:
    - Loads and preprocesses the image from the given path.
    - Loads the trained model.
    - Retrieves the class names.
    - Predicts the class label for the image.
    - Prints the image path and the predicted class name.

    Args:
        path (str): Path to the image file.
    """
    img = get_image(
        path,
        PREDICT_SAMPLE_CONFIG["image_size"],
        PREDICT_SAMPLE_CONFIG["image_channels"],
    )
    model = load_model(PREDICT_SAMPLE_CONFIG["model_path"])
    class_names = get_ds_class_names()
    pred_class_name = get_prediction_class_name(model, img, class_names)
    print(f"Image Path: {path}, Model Prediction: {pred_class_name}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    args.add_argument("--path", "-p", type=str)
    parsed_args = args.parse_args()

    predict_sample(parsed_args.path)
