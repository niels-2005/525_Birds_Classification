from get_data import get_training_datasets
from get_model import build_model
from get_training_callbacks import get_training_callbacks

MODEL_TRAINING_CONFIG = {"epochs": 50}


def train_model():
    """
    Trains a deep learning model using the specified training and validation datasets.

    The function:
    - Loads the training and validation datasets.
    - Builds the model.
    - Retrieves the necessary training callbacks.
    - Trains the model using the specified number of epochs and callbacks.

    Uses configurations from MODEL_TRAINING_CONFIG.

    Returns:
        None
    """
    train_dataset, valid_dataset = get_training_datasets()
    model = build_model()
    callbacks = get_training_callbacks()
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=MODEL_TRAINING_CONFIG["epochs"],
        callbacks=callbacks,
    )


if __name__ == "__main__":
    train_model()
