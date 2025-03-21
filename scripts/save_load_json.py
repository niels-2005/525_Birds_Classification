import json


def save_file_as_json(filepath, best_hparams):
    with open(filepath, "w") as json_file:
        json.dump(best_hparams, json_file, indent=4)


def load_json_from_file(filepath):
    with open(filepath, "r") as json_file:
        return json.load(json_file)
