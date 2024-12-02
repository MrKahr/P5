from typing import Any, Callable, Union
import keras
from numpy.random import RandomState
from scikeras.wrappers import KerasClassifier

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
        solver: str = "adam",  # "optimizer" in TensorFlow
        activation: str = "relu",
        loss: Union[str, Callable[..., Any], None] = None,
        learning_rate: str = "constant",
        learning_rate_init: float = 0.001,
        alpha: float = 0.0001,
        max_iter: int = 200,  # "epochs" in  TensorFlow
        tol: float = 0.0001,
        random_state: Union[int, RandomState, None] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.solver = solver
        self.activation = activation
        self.loss = loss
        self.epochs = max_iter

    def _keras_build_fn(self, compile_kwargs: dict[str, Any]):
        model = keras.Sequential()
        inp = keras.layers.Input(shape=(self.hidden_layer_sizes[0],))
        model.add(inp)

        # Will fail if no hidden layer is present
        for hidden_layer_size in self.hidden_layer_sizes[1:]:
            layer = keras.layers.Dense(hidden_layer_size, activation=self.activation)
            model.add(layer)

        if self.target_type_ == "binary":
            n_output_units = 1
            output_activation = "sigmoid"
            loss = "binary_crossentropy"
        elif self.target_type_ == "multiclass":
            n_output_units = self.n_classes_
            output_activation = "softmax"
            loss = "sparse_categorical_crossentropy"
        else:
            raise NotImplementedError(f"Unsupported task type: {self.target_type_}")
        out = keras.layers.Dense(n_output_units, activation=output_activation)
        model.add(out)
        model.compile(loss=loss, optimizer=compile_kwargs["optimizer"])
        return model
