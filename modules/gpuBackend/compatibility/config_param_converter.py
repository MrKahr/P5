from typing import Any

import numpy as np


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
        if "activation" in kwargs:
            if isinstance(kwargs["activation"], (list, np.ndarray)):
                arr = []
                for item in kwargs["activation"]:
                    if item == "logistic":
                        item = "sigmoid"
                    arr.append(item)
                kwargs["activation"] = np.array(arr)
            elif kwargs["activation"] == "logistic":
                kwargs["activation"] = "sigmoid"
        return kwargs
