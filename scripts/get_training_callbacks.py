import tensorflow as tf

CALLBACKS_CONFIG = {
    "tensorboard_log_dir": "../tb_training_logs_6",
    "model_save_path": "../trained_model_4/model.keras",
    "save_best_model_only": True,
    "metric_to_monitor": "val_loss",
    "early_stopping_patience": 9,
    "reduce_lr_patience": 6,
}


def get_tensorboard_callback():
    """
    Creates and returns a TensorBoard callback for logging training metrics.

    The callback:
    - Logs training metrics such as loss and accuracy for visualization in TensorBoard.
    - Saves logs in the directory specified in CALLBACKS_CONFIG.

    Returns:
        tf.keras.callbacks.TensorBoard: The TensorBoard callback instance.
    """
    return tf.keras.callbacks.TensorBoard(
        log_dir=CALLBACKS_CONFIG["tensorboard_log_dir"]
    )


def get_model_checkpoint_callback():
    """
    Creates and returns a ModelCheckpoint callback to save the best model during training.

    The callback:
    - Monitors a specified metric.
    - Saves the model only when the monitored metric improves.

    Returns:
        tf.keras.callbacks.ModelCheckpoint: The ModelCheckpoint callback instance.
    """
    return tf.keras.callbacks.ModelCheckpoint(
        CALLBACKS_CONFIG["model_save_path"],
        monitor=CALLBACKS_CONFIG["metric_to_monitor"],
        save_best_only=CALLBACKS_CONFIG["save_best_model_only"],
    )


def get_early_stopping_callback():
    """
    Creates and returns an EarlyStopping callback to prevent overfitting.

    The callback:
    - Monitors a specified metric.
    - Stops training when no improvement is observed for a defined number of epochs.

    Returns:
        tf.keras.callbacks.EarlyStopping: The EarlyStopping callback instance.
    """
    return tf.keras.callbacks.EarlyStopping(
        monitor=CALLBACKS_CONFIG["metric_to_monitor"],
        patience=CALLBACKS_CONFIG["early_stopping_patience"],
    )


def get_reduce_lr_on_plateau_callback():
    """
    Creates and returns a ReduceLROnPlateau callback to adjust the learning rate dynamically.

    The callback:
    - Monitors a specified metric.
    - Reduces the learning rate when no improvement is detected for a defined number of epochs.

    Returns:
        tf.keras.callbacks.ReduceLROnPlateau: The ReduceLROnPlateau callback instance.
    """
    return tf.keras.callbacks.ReduceLROnPlateau(
        monitor=CALLBACKS_CONFIG["metric_to_monitor"],
        patience=CALLBACKS_CONFIG["reduce_lr_patience"],
    )


def get_training_callbacks():
    """
    Creates and returns a list of training callbacks for monitoring and optimizing model training.

    The function initializes the following callbacks based on CALLBACKS_CONFIG:
    - TensorBoard: Logs training metrics for visualization.
    - ModelCheckpoint: Saves the best model based on the monitored metric.
    - EarlyStopping: Stops training if the monitored metric does not improve.
    - ReduceLROnPlateau: Reduces the learning rate when the monitored metric stagnates.

    Returns:
        list[tf.keras.callbacks.Callback]: A list of TensorFlow Keras callback instances.
    """
    tensorboard_callback = get_tensorboard_callback()
    model_checkpoint_callback = get_model_checkpoint_callback()
    early_stopping_callback = get_early_stopping_callback()
    reduce_lr_callback = get_reduce_lr_on_plateau_callback()

    return [
        tensorboard_callback,
        model_checkpoint_callback,
        early_stopping_callback,
        reduce_lr_callback,
    ]
