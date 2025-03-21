import tensorflow as tf


CALLBACKS_CONFIG = {
    "tensorboard_log_dir": "../tb_training_logs_6",
    "model_save_path": "../trained_model_4/model.keras",
    "save_best_model_only": True,
    "metric_to_monitor": "val_loss",
    "early_stopping_patience": 9,
    "reduce_lr_patience": 6,
}


def get_training_callbacks():
    """
    Creates and returns a list of training callbacks for monitoring and optimizing model training.

    The function initializes the following callbacks based on CALLBACKS_CONFIG:
    - TensorBoard: Logs training metrics for visualization.
    - ModelCheckpoint: Saves the best model based on the monitored metric.
    - EarlyStopping: Stops training if the monitored metric does not improve.
    - ReduceLROnPlateau: Reduces the learning rate when the monitored metric stagnates.

    Returns:
        list: A list of TensorFlow Keras callback instances.
    """
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=CALLBACKS_CONFIG["tensorboard_log_dir"]
    )
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        CALLBACKS_CONFIG["model_save_path"],
        monitor=CALLBACKS_CONFIG["metric_to_monitor"],
        save_best_only=CALLBACKS_CONFIG["save_best_model_only"],
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor=CALLBACKS_CONFIG["metric_to_monitor"],
        patience=CALLBACKS_CONFIG["early_stopping_patience"],
    )
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=CALLBACKS_CONFIG["metric_to_monitor"],
        patience=CALLBACKS_CONFIG["reduce_lr_patience"],
    )
    return [
        tensorboard_callback,
        model_checkpoint_callback,
        early_stopping_callback,
        reduce_lr_callback,
    ]

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