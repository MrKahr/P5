from typing import Callable, Union
from keras import activations


class ActivationFunctionSelector:
    # Get the function pointer to activation function 

    @classmethod
    def _getSigmoid(cls) -> activations.sigmoid:
        return activations.sigmoid

    @classmethod
    def _getRelu(cls) -> activations.relu:
        return activations.relu
    @classmethod
    def _getTanh(cls) -> activations.tanh:
        return activations.tanh

    @classmethod
    def getActivationFunction(
        cls, activationFunction: str | Callable
    ) -> Union[activations.sigmoid, activations.relu,activations.tanh]:
        # If already defined in keras, don't waster operations searching for it
        if callable(activationFunction):
            return activationFunction

        if activationFunction == "logistic":
            # Different naming in kera caused issue where NN could not be fitted.
            activationFunction = cls._getSigmoid()
        elif activationFunction == "relu":
            activationFunction = cls._getRelu()
        elif activationFunction == "tanh":
            activationFunction = cls._getTanh()
        else:
            raise ValueError(
                f"Invalid activation function '{activationFunction}'. Expected one of [sigmoid, relu, tanh]"
            )
        return activationFunction
