from typing import Union
from keras import regularizers


class RegularizerSelector:

    @classmethod
    def _getL2(cls, **kwargs) -> regularizers.L2:
        # Values taken from scikit-learns default values of MLPClassifier
        return regularizers.L2(**kwargs)

    @classmethod
    def getRegularizer(
        cls, regularizer: str | object, **kwargs
    ) -> Union[regularizers.L2]:
        if isinstance(regularizer, (regularizers.L2)):
            return regularizer

        if regularizer == "l2":
            regularizer = cls._getL2(**kwargs)
        else:
            raise ValueError(
                f"Invalid solver '{regularizer}'. Expected one of [adam, sgd]"
            )
        return regularizer