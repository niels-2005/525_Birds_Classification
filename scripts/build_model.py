import tensorflow as tf

EFFICIENTNET_CONFIG = {
    "include_top": False,
    "input_shape": (224, 224, 3),
    "pooling": "avg",
    "trainable": True,
}

MODEL_LAYERS_CONFIG = {
    "data_augmentation_flip": tf.keras.layers.RandomFlip("horizontal"),
    "l2_regularization": tf.keras.regularizers.L2(0.01),
    "output_units": 525,
    "output_activation": "softmax",
}

MODEL_COMPILE_CONFIG = {
    "optimizer": tf.keras.optimizers.Adam(learning_rate=1e-4),
    "loss": "categorical_crossentropy",
    "metric": ["accuracy"],
}


def build_efficientnet_model():
    """
    Builds and returns an EfficientNetB0 model with configurations specified in EFFICIENTNET_CONFIG.

    The function:
    - Initializes an EfficientNetB0 model with predefined configurations.
    - Sets whether the model's weights should be trainable.

    Returns:
        tf.keras.Model: The EfficientNetB0 model configured according to EFFICIENTNET_CONFIG.
    """
    efficient_net = tf.keras.applications.EfficientNetB0(
        include_top=EFFICIENTNET_CONFIG["include_top"],
        input_shape=EFFICIENTNET_CONFIG["input_shape"],
        pooling=EFFICIENTNET_CONFIG["pooling"],
    )
    efficient_net.trainable = EFFICIENTNET_CONFIG["trainable"]
    return efficient_net


def add_model_layers(model, efficient_net):
    """
    Adds layers to a given model, including data augmentation, EfficientNet as a feature extractor,
    and a dense output layer.

    The function:
    - Adds a data augmentation layer.
    - Incorporates EfficientNetB0 as a feature extractor.
    - Appends a dense output layer with specified activation and regularization.

    Args:
        model (tf.keras.Sequential): The base model to which layers will be added.
        efficient_net (tf.keras.Model): The EfficientNetB0 model used for feature extraction.

    Returns:
        tf.keras.Sequential: The updated model with added layers.
    """
    model.add(MODEL_LAYERS_CONFIG["data_augmentation_flip"])
    model.add(efficient_net)
    model.add(
        tf.keras.layers.Dense(
            units=MODEL_LAYERS_CONFIG["output_units"],
            activation=MODEL_LAYERS_CONFIG["output_activation"],
            kernel_regularizer=MODEL_LAYERS_CONFIG["l2_regularization"],
        )
    )
    return model


def build_model():
    """
    Builds and compiles a deep learning model using EfficientNetB0 as a feature extractor.

    The function:
    - Creates a Sequential model.
    - Initializes EfficientNetB0 with predefined configurations.
    - Adds necessary layers including data augmentation and output layers.
    - Compiles the model with the specified optimizer, loss function, and metrics.

    Returns:
        tf.keras.Model: The compiled deep learning model.
    """
    model = tf.keras.Sequential()
    efficient_net = build_efficientnet_model()
    model = add_model_layers(model, efficient_net)
    model.compile(
        optimizer=MODEL_COMPILE_CONFIG["optimizer"],
        loss=MODEL_COMPILE_CONFIG["loss"],
        metrics=MODEL_COMPILE_CONFIG["metric"],
    )
    return model
