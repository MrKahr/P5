from typing import Any, Callable
from scikeras.wrappers import KerasClassifier


# Docs: https://adriangb.com/scikeras/stable/
# DOCS: https://www.tensorflow.org/api_docs/python/tf/keras/Model#by_subclassing_the_model_class
# Docs: https://intel.github.io/scikit-learn-intelex/latest/algorithms.html / https://github.com/intel/scikit-learn-intelex
class MLPClassifierGPU(KerasClassifier):
    # Docs: https://adriangb.com/scikeras/stable/notebooks/MLPClassifier_MLPRegressor.html
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        optimizer: str = "adam",
        loss: str | Callable[..., Any] | None = None,
        epochs: int = 200,
        **kwargs
    ):
        super().__init__(**kwargs)
