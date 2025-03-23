import tensorflow as tf


def load_model(model_path: str):
    """
    Loads a trained TensorFlow Keras model from a given file path.

    This function is used to restore a previously saved model for inference or further training.

    Args:
        model_path (str): The file path where the model is stored.

    Returns:
        tf.keras.Model: The loaded model ready for predictions or retraining.
    """
    model = tf.keras.models.load_model(model_path)
    return model
