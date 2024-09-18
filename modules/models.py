import os
from pathlib import Path
import pandas as pd
import sklearn
import math
from sklearn.naive_bayes import GaussianNB
from dataPreprocessing import DataProcessor

# TODO: First model is Naive Bayes. We'll try the Gaussian Naive Bayes
# https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes


class GaussianNaiveBayes:
    def __init__(self, data: pd.DataFrame) -> None:
        # Data must be split in test and training sets - halves chosen arbitrarily
        self.trainingData = data["id" <= (math.floor(data.size() * 0.5))]
        self.testData = data["id" <= math.ceil(data.size() * 0.5)]

    def generatePrediction(self, X_train, y_train, X_test, y_test) -> None:
        gnb = GaussianNB()
        y_pred = gnb.fit(X_train, y_train).predict(X_test)
        print(
            "Number of mislabeled points out of a total %d points : %d"
            % (X_test.shape[0], (y_test != y_pred).sum())
        )  # from https://scikit-learn.org/stable/modules/naive_bayes.html


# Process data
dp = DataProcessor(
    Path(
        Path(os.path.split(__file__)[0]).parents[0],
        "data/eksperimentelle_sår_2024_regs.csv",
    )
)
dp.process()

# Model data
model = GaussianNaiveBayes(dp.dataFrame)
# Get prediction from training and test sets
model.generatePrediction(
    model.trainingData["Kontraktion", "Hyperæmki", "Ødem", "Eksudat"],
    model.trainingData["Infektionsniveau"],
    model.testData["Kontraktion", "Hyperæmki", "Ødem", "Eksudat"],
    model.testData["Infektionsniveau"],
)
