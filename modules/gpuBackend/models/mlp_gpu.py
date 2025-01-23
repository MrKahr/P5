import keras
from numpy.random import RandomState
from scikeras.wrappers import KerasClassifier
from typing import Any, Callable, Literal, Union, override
from math import ceil

from modules.gpuBackend.activation.activationFunctionSelector import (
    ActivationFunctionSelector,
)
from modules.gpuBackend.optimizers.optimizerSelector import OptimizerSelector
from modules.gpuBackend.regularizer.regularizerSelector import RegularizerSelector

# SECTION - SciKeras MLPClassifier example
# https://adriangb.com/scikeras/stable/notebooks/MLPClassifier_MLPRegressor.html

# SECTION - Tensorflow Model docs
# https://www.tensorflow.org/api_docs/python/tf/keras/Model#by_subclassing_the_model_class

# SECTION - Solvers (optimizers in Keras / TensorFlow)
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers

# SECTION - Activation function
# https://www.tensorflow.org/api_docs/python/tf/keras/activations

# SECTION - Loss function
# https://www.tensorflow.org/api_docs/python/tf/keras/losses


class MLPClassifierGPU(KerasClassifier):
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        optimizer: (
            Literal["adam", "sgd"] | keras.Optimizer
        ) = "adam",  # "solver" in scikit
        activation: Literal["sigmoid", "relu", "tanh"] | Callable = "relu",
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        alpha: float = 0.0001,
        epochs: int = 200,  # "max_iter" in scikit
        batch_size: Union[int, None] = 200,
        tol: float = 0.0001,
        n_iter_no_change: int = 10,
        random_state: Union[int, RandomState, None] = None,
        verbose: int = 1,
        callbacks=None,
        **kwargs,  # Needed to remove garbage arguments from Keras
    ):
        """
        A CUDA GPU-based Multi-Layer Perceptron implementing scikit-learns' estimator API.

        Parameters
        ----------
        hidden_layer_sizes : tuple, optional
            Each element in the tuple represent a layer in the neural network,
            where the element's value represent the number of neurons for that layer.
            By default `(100,)`.

        optimizer : Literal["adam", "sgd"] | keras.Optimizer, optional
            The algorithm used to optimize weights.
            Can be an instance of a keras optimizer or a string literal.
            By default `"adam"`.
            NOTE: Called "solver" in scikit-learn.

        activation: Literal["logistic","relu","tanh"] | Callable
            Activation function for all layers except input/output.
            By default "relu".

        learning_rate : str, optional
            Adjust learning rate across epochs.
            By default "constant".
            NOTE: Included for compatibiity with scikit-learn but currently not implemented.

        learning_rate_init : float, optional
            Learning rate for weigth updates.
            By default `0.001`.

        alpha : float, optional
            Strength of the L2 regularization term.
            By default `0.0001`.

        epochs : int, optional
            Number of training iterations.
            By default `200`.
            NOTE: Called "max_iter" in scikit-learn.

        batch_size : int, optional
            Size of minibatches for stochastic optimizers.
            By default `200`.

        tol : float
            Tolerance for the optimization.
            When the loss or score is not improving by at least tol for '5'
            consecutive iterations, convergence is considered to be reached and training stops.
            By default `0.0001`.

        n_iter_no_change : int
            Maximum number of epochs to not meet `tol` improvement.
            By default `10`.

        random_state : Union[int, RandomState, None], optional
            Determines random number generation for weights and bias initialization etc.
            Pass an int for reproducible results across multiple function calls.
            By default `None`.

        verbose : int, optional
            Enable verbose logging of training and inference.
            By default `1`.
        """
        self.callbacks = callbacks or [  # Workaround for scikit API
            keras.callbacks.EarlyStopping(
                monitor="loss",  # Monitor loss function
                mode="min",  # How to interpret value ("min" == minimize)
                min_delta=tol,  # Must improve greater than this
                patience=n_iter_no_change,  # Improvement not >= "min_delta" after X consecutive epochs
                baseline=2,  # Value must be "min" (less) than this to continue (after "start_from_epoch")
                start_from_epoch=min(
                    0.1 * epochs, 250
                ),  # Start monitoring after X epochs
                verbose=verbose,
            ),
        ]
        if learning_rate == "adaptive":
            self.callbacks.append(
                keras.callbacks.ReduceLROnPlateau(
                    monitor="loss",
                    factor=0.1,  # Reduce learning rate by factor
                    patience=ceil(
                        n_iter_no_change / 2
                    ),  # Improvement not >= "min_delta" after X consecutive epochs
                    mode="min",
                    min_delta=0.0001,
                    min_lr=0.00001,
                ),
            )

        super().__init__(
            optimizer=OptimizerSelector.getOptimizer(
                optimizer, learning_rate=learning_rate_init
            ),
            batch_size=batch_size,
            random_state=random_state,
            callbacks=self.callbacks,
            epochs=epochs,
            verbose=verbose,
        )
        self.activation = ActivationFunctionSelector.getActivationFunction(activation)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.alpha = alpha
        self.tol = tol
        self.epochs = epochs
        self.n_iter_no_change = n_iter_no_change

    def _keras_build_fn(self, compile_kwargs: dict[str, Any]):
        """
        Build and compile the underlying Keras model.

        Parameters
        ----------
        compile_kwargs : dict[str, Any]
            kwargs supplied by Keras.

        Returns
        -------
        keras.Sequential
            The compiled Keras model.

        Raises
        ------
        NotImplementedError
            The classification type is not supported.
        """
        model = keras.Sequential()
        # Input layer
        input = keras.layers.Input(shape=(self.n_features_in_,))
        model.add(input)

        # Hidden layers
        for hidden_layer_size in self.hidden_layer_sizes:
            layer = keras.layers.Dense(
                hidden_layer_size,
                activation=self.activation,
                kernel_regularizer=RegularizerSelector.getRegularizer(
                    "l2", l2=self.alpha
                ),
            )
            model.add(layer)

        # Output layer
        if self.target_type_ == "binary":
            n_output_units = 1

            # Sigmoid is used as the activation for the last layer of the
            # classification network because the result is binary.
            output_activation = "sigmoid"
            loss = keras.losses.BinaryCrossentropy()  # Log-loss function
        elif self.target_type_ == "multiclass":
            n_output_units = self.n_classes_  # Number of target labels to predict

            # Softmax is used as the activation for the last layer of the
            # classification network because the result can be interpreted as
            # a probability distribution.
            # https://www.tensorflow.org/api_docs/python/tf/keras/activations/softmax
            output_activation = "softmax"
            loss = (
                keras.losses.SparseCategoricalCrossentropy()
            )  # Multiclass log-loss function
        else:
            raise NotImplementedError(f"Unsupported task type: {self.target_type_}")

        out = keras.layers.Dense(
            n_output_units,
            activation=output_activation,
            kernel_regularizer=RegularizerSelector.getRegularizer("l2", l2=self.alpha),
        )
        model.add(out)
        model.compile(loss=loss, optimizer=compile_kwargs["optimizer"])
        return model

    @override
    def fit(self, X, y, sample_weight=None, **kwargs):
        if self.verbose == 0:
            keras.config.disable_interactive_logging()
            local_verbose = 1
        else:
            local_verbose = self.verbose

        return super().fit(X, y, sample_weight, verbose=local_verbose, **kwargs)
