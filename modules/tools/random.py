from typing import override
import numpy as np


class RNG(np.random.RandomState):
    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed)

    @override
    def randint(
        self,
        low: int,
        high: int | None = None,
    ) -> np.int64:
        return super().randint(low, high, dtype=np.int64)
