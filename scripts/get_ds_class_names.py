import json
import os

from get_data import get_test_dataset

CLASS_NAMES_CONFIG = {"class_names_save_path": "../class_names.json"}


def save_class_names_as_json(class_names):
    """
    Saves the provided class names to a JSON file.

    The function writes the class names to a file specified in CLASS_NAMES_CONFIG["class_names_save_path"].

    Args:
        class_names (list): A list of class names to be saved.
    """
    with open(CLASS_NAMES_CONFIG["class_names_save_path"], "w") as f:
        json.dump(class_names, f)


def load_class_names(path):
    """
    Loads class names from a JSON file.

    Args:
        path (str): Path to the JSON file containing class names.

    Returns:
        list: A list of class names.
    """
    with open(path, "r") as f:
        return json.load(f)


def get_ds_class_names():
    """
    Retrieves the class names from the dataset or loads them from a saved file.

    The function:
    - Checks if the class names JSON file exists.
    - If the file exists, it loads and returns the class names.
    - If the file does not exist, it extracts class names from the test dataset, saves them, and then returns them.

    Returns:
        list: A list of class names.
    """
    if not os.path.exists(CLASS_NAMES_CONFIG["class_names_save_path"]):
        test_dataset = get_test_dataset()
        class_names = test_dataset.class_names
        save_class_names_as_json(class_names)
        return class_names
    else:
        return load_class_names(CLASS_NAMES_CONFIG["class_names_save_path"])
