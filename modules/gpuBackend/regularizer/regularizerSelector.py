from typing import Union
from keras import regularizers


class RegularizerSelector:

    @classmethod
    def _getL2(cls, **kwargs) -> regularizers.L2:
        return regularizers.L2(**kwargs)

    @classmethod
    def getRegularizer(cls, regularizer: str, **kwargs) -> Union[regularizers.L2]:
        # FIXME: Bad workaround
        if "l2" in kwargs and isinstance(kwargs["l2"], (regularizers.L2)):
            return regularizer

        if regularizer == "l2":
            regularizer = cls._getL2(**kwargs)
        else:
            raise ValueError(
                f"Invalid regularizer '{regularizer}'. Expected one of [l2]"
            )
        return regularizer
