from typing import Any, Callable, Union
import keras
from numpy.random import RandomState
from scikeras.wrappers import KerasClassifier

from modules.gpuBackend.optimizers.optimizerSelector import OptimizerSelector
from modules.logging import logger

# SECTION - Intel Scikit-Learn Optimization
# https://github.com/intel/scikit-learn-intelex
# Supported models: https://intel.github.io/scikit-learn-intelex/latest/algorithms.html


# SECTION - SciKeras MLPClassifier example
# https://adriangb.com/scikeras/stable/notebooks/MLPClassifier_MLPRegressor.html

# SECTION - Tensorflow Model docs
# https://www.tensorflow.org/api_docs/python/tf/keras/Model#by_subclassing_the_model_class


# SECTION - Score Metrics
# https://www.tensorflow.org/api_docs/python/tf/keras/Metric#example
# https://www.tensorflow.org/api_docs/python/tf/keras/metrics


# SECTION - solvers (optimizers in TensorFlow)
# https://www.tensorflow.org/api_docs/python/tf/keras/optimizers

# SECTION - loss function
# https://www.tensorflow.org/api_docs/python/tf/keras/losses


class MLPClassifierGPU(KerasClassifier):
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        solver: str = "adam",  # "solver" in scikit
        activation: str = "relu",
        loss: Union[str, Callable[..., Any], None] = None,
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        alpha: float = 0.0001,
        max_iter: int = 200,  # "epochs" in  TensorFlow
        tol: float = 0.0001,
        scikit_compat: bool = True,
        random_state: Union[int, RandomState, None] = None,
        **kwargs,
    ):
        super().__init__(
            optimizer=OptimizerSelector.getOptimizer(
                solver, learning_rate=learning_rate_init
            ),
            loss=loss,
            random_state=random_state,
            epochs=max_iter,
            **kwargs,
        )
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.scikit_compat = scikit_compat

        if learning_rate != "constant":
            logger.warning(
                f"Only 'constant' learning rate is supported, switching to 'constant'. Got '{learning_rate}'"
            )

    def _keras_build_fn(self, compile_kwargs: dict[str, Any]):
        model = keras.Sequential()
        # Input layer
        inp = keras.layers.Input(shape=(self.hidden_layer_sizes[0],))
        model.add(inp)

        # Hidden layers
        if len(self.hidden_layer_sizes) > 2:
            for hidden_layer_size in self.hidden_layer_sizes[1:-2]:
                layer = keras.layers.Dense(
                    hidden_layer_size, activation=self.activation
                )
                model.add(layer)

        # Output layer
        if self.scikit_compat:
            output_activation = self.activation
            n_output_units = self.hidden_layer_sizes[-1]
            loss = compile_kwargs["loss"]
        else:
            if self.target_type_ == "binary":
                n_output_units = 1
                output_activation = "sigmoid"
                loss = "binary_crossentropy"  # Log-loss function
            elif self.target_type_ == "multiclass":
                n_output_units = self.n_classes_  # n_outputs_expected_
                output_activation = "softmax"
                loss = "categorical_crossentropy "  # Multiclass log-loss function
            else:
                raise NotImplementedError(f"Unsupported task type: {self.target_type_}")

        out = keras.layers.Dense(n_output_units, activation=output_activation)
        model.add(out)
        model.compile(loss=loss, optimizer=compile_kwargs["optimizer"])
        return model
