import tensorflow as tf


def get_prediction_class_name(model, img, class_names):
    """
    Predicts the class label for a given image using the provided model.

    The function:
    - Expands the image dimensions to match the model input shape.
    - Gets the prediction probabilities from the model.
    - Finds the class index with the highest probability.
    - Maps the index to the corresponding class name.

    Args:
        model (tf.keras.Model): The trained model used for prediction.
        img (tf.Tensor): The preprocessed image tensor.
        class_names (list): A list of class names corresponding to model output indices.

    Returns:
        str: The predicted class name.
    """
    y_probs = model.predict(tf.expand_dims(img, axis=0))
    y_pred = tf.argmax(y_probs, axis=1).numpy()[0]
    return class_names[y_pred]
