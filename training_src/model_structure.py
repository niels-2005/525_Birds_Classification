import tensorflow as tf

EFFICIENTNET_CONFIG = {
    "include_top": False,
    "input_shape": (224, 224, 3),
    "pooling": "avg",
    "trainable": True,
}

MODEL_LAYERS_CONFIG = {
    "data_augmentation_flip": tf.keras.layers.RandomFlip("horizontal"),
    "output_units": 525,
    "output_activation": "softmax",
}

MODEL_COMPILE_CONFIG = {
    "optimizer": tf.keras.optimizers.Adam(learning_rate=1e-4),
    "loss": "categorical_crossentropy",
    "metric": ["accuracy"],
}