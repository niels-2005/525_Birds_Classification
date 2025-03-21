import tensorflow as tf 


def load_model(model_path: str):
    """
    Loads a pre-trained TensorFlow Keras model from the specified file path.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        tf.keras.Model: The loaded Keras model.
    """
    model = tf.keras.models.load_model(model_path)
    return model