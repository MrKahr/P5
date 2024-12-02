from keras import optimizers


class OptimizerSelector:
    def __init__(self) -> None:
        pass

    def _getAdam(self, **kwargs) -> optimizers.Adam:
        # Values taken from scikit-learns default values of MLPClassifier
        return optimizers.Adam(epsilon=1e-08, **kwargs)

    def _getSGD(self, **kwargs) -> optimizers.SGD:
        # Values taken from scikit-learns default values of MLPClassifier
        return optimizers.SGD(nesterov=True, momentum=0.9, **kwargs)
