import pandas as pd
import math
from sklearn.naive_bayes import GaussianNB


class GaussianNaiveBayes:
    def __init__(self, data: pd.DataFrame) -> None:
        # Data must be split in test and training sets - halves chosen arbitrarily
        self.trainingData = data[data["Gris ID"] <= (math.floor(data.size * 0.5))]
        self.testData = data[data["Gris ID"] <= math.ceil(data.size * 0.5)]
        self.xp = None  # type: int
        self.yp = None  # type: int

    def generatePrediction(self, X_train, y_train, X_test, y_test) -> None:
        # from https://scikit-learn.org/stable/modules/naive_bayes.html
        gnb = GaussianNB()
        y_pred = gnb.fit(X_train, y_train).predict(X_test)

        self.xp = X_test.shape[0]  # type: int
        self.yp = (y_test != y_pred).sum()  # type: int
        print(
            f"Number of mislabeled points out of a total {self.xp} points: {self.yp} (accuracy: {1-(self.yp/self.xp):.3f})"
        )

    def getResults(self) -> tuple[int]:
        return (self.xp, self.yp)
