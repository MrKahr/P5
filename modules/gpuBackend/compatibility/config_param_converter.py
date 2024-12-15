from typing import Any


class ConfigParamConverter:

    @classmethod
    def convertToMLPClassifierGPU(cls, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Convert kwargs from scikit to Keras terminology.
        E.g. "solver" in scikit is called "optimizer" in Keras.

        Parameters
        ----------
        kwargs : dict[str, Any]
            The kwargs to convert.

        Returns
        -------
        dict[str, Any]
            The converted kwargs.
        """
        if "solver" in kwargs:
            kwargs |= {"optimizer": kwargs.pop("solver")}

        if "max_iter" in kwargs:
            kwargs |= {"epochs": kwargs.pop("max_iter")}
        if "alpha" in kwargs:
            kwargs |= {"regularizer": kwargs.pop("alpha")}
        return kwargs
