from tensorflow.keras.preprocessing import image_dataset_from_directory  # type: ignore


DATA_CONFIG = {
    "train_path": "../dataset/train",
    "test_path": "../dataset/test",
    "label_mode": "categorical",
    "seed": 42,
    "image_size": (224, 224),
    "batch_size": 32,
    "shuffle_on_train": True,
    "shuffle_on_test": False,
    "validation_split": 0.2,
    "training_subset": "training",
    "validation_subset": "validation",
}


def get_training_datasets():
    """
    Loads and returns the training and validation datasets from the specified directory.

    The function:
    - Loads the training dataset with configurations from DATA_CONFIG.
    - Loads the validation dataset using the same configurations.
    - Uses `image_dataset_from_directory` to create TensorFlow datasets.

    Returns:
        tuple: (train_dataset, valid_dataset), where:
            - train_dataset (tf.data.Dataset): Training dataset.
            - valid_dataset (tf.data.Dataset): Validation dataset.
    """
    train_dataset = image_dataset_from_directory(
        DATA_CONFIG["train_path"],
        label_mode=DATA_CONFIG["label_mode"],
        seed=DATA_CONFIG["seed"],
        image_size=DATA_CONFIG["image_size"],
        batch_size=DATA_CONFIG["batch_size"],
        shuffle=DATA_CONFIG["shuffle_on_train"],
        validation_split=DATA_CONFIG["validation_split"],
        subset=DATA_CONFIG["training_subset"],
    )

    valid_dataset = image_dataset_from_directory(
        DATA_CONFIG["train_path"],
        label_mode=DATA_CONFIG["label_mode"],
        seed=DATA_CONFIG["seed"],
        image_size=DATA_CONFIG["image_size"],
        batch_size=DATA_CONFIG["batch_size"],
        shuffle=DATA_CONFIG["shuffle_on_train"],
        validation_split=DATA_CONFIG["validation_split"],
        subset=DATA_CONFIG["validation_subset"],
    )
    return train_dataset, valid_dataset


def get_test_dataset():
    """
    Loads and returns the test dataset from the specified directory.

    The function:
    - Loads the test dataset using `image_dataset_from_directory`.
    - Uses configurations from DATA_CONFIG to set parameters such as 
      image size, batch size, and whether shuffling is applied.

    Returns:
        tf.data.Dataset: The test dataset.
    """
    test_dataset = image_dataset_from_directory(
        DATA_CONFIG["test_path"],
        label_mode=DATA_CONFIG["label_mode"],
        seed=DATA_CONFIG["seed"],
        image_size=DATA_CONFIG["image_size"],
        batch_size=DATA_CONFIG["batch_size"],
        shuffle=DATA_CONFIG["shuffle_on_test"],
    )
    return test_dataset
