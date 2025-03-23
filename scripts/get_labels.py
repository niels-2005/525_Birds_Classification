import tensorflow as tf


def get_predicted_labels(model, test_dataset):
    """
    Generates predicted class labels for a given test dataset using the provided model.

    The function:
    - Uses the trained model to make predictions on the test dataset.
    - Computes the class label with the highest probability for each sample.

    Args:
        model (tf.keras.Model): A trained model capable of making predictions.
        test_dataset (tf.data.Dataset): The dataset to be evaluated, typically a TensorFlow dataset
                                        containing images and (optionally) labels.

    Returns:
        tf.Tensor: A tensor containing the predicted class labels.
    """
    y_probs = model.predict(test_dataset)
    y_pred = tf.argmax(y_probs, axis=1)
    return y_pred


def get_actual_labels(test_dataset):
    """
    Extracts the ground truth labels from a given test dataset.

    The function:
    - Iterates over the dataset to retrieve the actual labels.
    - Concatenates them into a single tensor.
    - Converts one-hot encoded labels (if applicable) into class indices.

    Args:
        test_dataset (tf.data.Dataset): The dataset containing features and corresponding true labels.

    Returns:
        tf.Tensor: A tensor containing the actual class labels.
    """
    y_true = tf.concat([y for x, y in test_dataset], axis=0)
    y_true = tf.argmax(y_true, axis=1)
    return y_true
