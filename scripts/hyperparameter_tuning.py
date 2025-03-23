import keras_tuner
import tensorflow as tf
from get_data import get_training_datasets


class HyperparameterTuner:
    """
    A class for hyperparameter tuning of an EfficientNet-based deep learning model.

    This class utilizes Keras Tuner for hyperparameter optimization,
    including data augmentation options, model architecture, and training configurations.

    Attributes:
        Various hyperparameter configurations related to the EfficientNet model,
        data augmentation, model architecture, optimizer, and tuning process.
    """

    def __init__(self):
        """
        Initializes the HyperparameterTuner with predefined hyperparameter ranges
        and configurations for model tuning.
        """
        self.tensorboard_log_dir = "../tb_logs"
        self.efficientnet_input_shape = (224, 224, 3)
        self.efficientnet_include_top = False
        self.efficientnet_trainable = False
        self.data_augmentation_rotation = 0.15
        self.data_augmentation_brightness = 0.15
        self.efficientnet_pooling = ["avg"]
        self.model_hidden_layers_range = [1, 2]
        self.model_hidden_units_range = [64, 512]
        self.model_hidden_activation = "relu"
        self.model_dropout_range = [0.1, 0.5]
        self.model_output_units = 525
        self.model_output_activation = "softmax"
        self.model_optimizer_choices = ["adam", "rmsprop"]
        self.model_loss = "categorical_crossentropy"
        self.model_metric = "accuracy"
        self.tuner_objective = "val_accuracy"
        self.tuner_max_trials = 30
        self.tuner_executions_per_trial = 1
        self.tuner_overwrite = True
        self.tuner_directory = "../hparam_tuning"
        self.tuner_project_name = "525_birds_classifer"
        self.tuner_epochs_per_trail = 8

    def build_efficientnet_model(self):
        """
        Builds an EfficientNetB0 model with predefined configurations.

        Returns:
            tf.keras.Model: The EfficientNetB0 model with the specified settings.
        """
        efficient_net = tf.keras.applications.EfficientNetB0(
            include_top=self.efficientnet_include_top,
            input_shape=self.efficientnet_input_shape,
            pooling="avg",
        )
        efficient_net.trainable = self.efficientnet_trainable
        return efficient_net

    def build_model(self, hp):
        """
        Builds a sequential model with EfficientNetB0 as a base and additional hyperparameter-tuned layers.

        Args:
            hp (keras_tuner.HyperParameters): The hyperparameter search space.

        Returns:
            tf.keras.Model: The compiled model with optimized hyperparameters.
        """
        model = tf.keras.Sequential()

        if hp.Boolean("add_random_flip"):
            model.add(tf.keras.layers.RandomFlip("horizontal"))

        if hp.Boolean("add_random_rotation"):
            model.add(
                tf.keras.layers.RandomRotation(factor=self.data_augmentation_rotation)
            )

        if hp.Boolean("add_random_brightness"):
            model.add(
                tf.keras.layers.RandomBrightness(
                    factor=self.data_augmentation_brightness
                )
            )

        efficient_net = self.build_efficientnet_model()
        model.add(efficient_net)

        for i in range(
            hp.Int(
                "num_layers",
                self.model_hidden_layers_range[0],
                self.model_hidden_layers_range[1],
            )
        ):
            model.add(
                tf.keras.layers.Dense(
                    units=hp.Int(
                        f"dense_units_{i}",
                        min_value=self.model_hidden_units_range[0],
                        max_value=self.model_hidden_units_range[1],
                        step=32,
                    ),
                    activation=self.model_hidden_activation,
                )
            )

            if hp.Boolean(f"enable_dropout_{i}"):
                model.add(
                    tf.keras.layers.Dropout(
                        rate=hp.Float(
                            f"dropout_rate_{i}",
                            min_value=self.model_dropout_range[0],
                            max_value=self.model_dropout_range[1],
                            step=0.1,
                        )
                    )
                )

        model.add(
            tf.keras.layers.Dense(
                self.model_output_units, activation=self.model_output_activation
            )
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=self.model_loss,
            metrics=[self.model_metric],
        )
        return model

    def get_tuner(self):
        """
        Initializes and returns a Keras Tuner RandomSearch object for hyperparameter optimization.

        Returns:
            keras_tuner.RandomSearch: The tuner configured for hyperparameter search.
        """
        self.build_model(keras_tuner.HyperParameters())
        return keras_tuner.RandomSearch(
            hypermodel=self.build_model,
            objective=self.tuner_objective,
            max_trials=self.tuner_max_trials,
            executions_per_trial=self.tuner_executions_per_trial,
            overwrite=self.tuner_overwrite,
            directory=self.tuner_directory,
            project_name=self.tuner_project_name,
        )

    def get_datasets(self):
        """
        Retrieves the training and validation datasets.

        Returns:
            tuple: A tuple containing the training and validation datasets.
        """
        train_dataset, valid_dataset = get_training_datasets()
        return train_dataset, valid_dataset

    def get_tensorboard_callback(self):
        """
        Creates and returns a TensorBoard callback for logging training metrics.

        Returns:
            tf.keras.callbacks.TensorBoard: The TensorBoard callback instance.
        """
        return tf.keras.callbacks.TensorBoard(log_dir=self.tensorboard_log_dir)

    def start_tuner_search(self):
        """
        Runs the hyperparameter tuning process.

        The function:
        - Initializes the tuner.
        - Loads training and validation datasets.
        - Sets up the TensorBoard callback.
        - Executes the search for optimal hyperparameters.

        Returns:
            None
        """
        tuner = self.get_tuner()
        train_dataset, valid_dataset = self.get_datasets()
        tensorboard_callback = self.get_tensorboard_callback()
        tuner.search(
            train_dataset,
            validation_data=valid_dataset,
            epochs=self.tuner_epochs_per_trail,
            callbacks=[tensorboard_callback],
        )
        self.save_best_hyperparameters(tuner)


if __name__ == "__main__":
    hparam_tuner = HyperparameterTuner()
    hparam_tuner.start_tuner_search()
