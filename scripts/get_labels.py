import tensorflow as tf


def get_predicted_labels(model, test_dataset):
    """
    Generates predicted labels for a given test dataset using the provided model.

    Args:
        model: A trained model capable of making predictions.
        test_dataset: The dataset to be evaluated, typically a TensorFlow dataset.

    Returns:
        tf.Tensor: Predicted class labels for the test dataset.
    """
    y_probs = model.predict(test_dataset)
    y_pred = tf.argmax(y_probs, axis=1)
    return y_pred


def get_actual_labels(test_dataset):
    """
    Extracts the ground truth labels from a given test dataset.

    Args:
        test_dataset: The dataset containing features and corresponding true labels.

    Returns:
        tf.Tensor: The actual class labels from the test dataset.
    """
    y_true = tf.concat([y for x, y in test_dataset], axis=0)
    y_true = tf.argmax(y_true, axis=1)
    return y_true
