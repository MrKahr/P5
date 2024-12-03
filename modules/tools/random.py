from typing import override
import numpy as np


class RNG(np.random.RandomState):
    def __init__(self, seed: int | None = None) -> None:
        """
        A subclass of np.random.RandomState where the
        method `randint` returns 64-bit integers per default.

        Parameters
        ----------
        seed : int | None, optional
            Random seed used to initialize the pseudo-random number generator.
            By default None.
        """
        super().__init__(seed)

    @override
    def randint(
        self,
        low: int,
        high: int | None = None,
    ) -> np.int64:
        return super().randint(low, high, dtype=np.int64)
