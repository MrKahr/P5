from typing import Union
from keras import optimizers


class OptimizerSelector:

    @classmethod
    def _getAdam(cls, **kwargs) -> optimizers.Adam:
        # Values taken from scikit-learns default values of MLPClassifier
        return optimizers.Adam(epsilon=1e-08, **kwargs)

    @classmethod
    def _getSGD(cls, **kwargs) -> optimizers.SGD:
        # Values taken from scikit-learns default values of MLPClassifier
        return optimizers.SGD(nesterov=True, momentum=0.9, **kwargs)

    @classmethod
    def getOptimizer(
        cls, optimizer: str | object, **kwargs
    ) -> Union[optimizers.Adam, optimizers.SGD]:
        if isinstance(optimizer, (optimizers.Adam, optimizers.SGD)):
            return optimizer

        if optimizer == "adam":
            optimizer = cls._getAdam(**kwargs)
        elif optimizer == "sgd":
            optimizer = cls._getSGD(**kwargs)
        else:
            raise ValueError(
                f"Invalid solver '{optimizer}'. Expected one of [adam, sgd]"
            )
        return optimizer
