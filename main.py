# Main entry-point into our code base
import os
from pathlib import Path

from modules.dataPreprocessing import DataProcessor
from modules.models import GaussianNaiveBayes
from modules.visualization import AccuracyPlotter

# Process data
dp = DataProcessor(
    Path(
        Path(os.path.split(__file__)[0]),
        "data/eksperimentelle_sår_2024_regs.csv",
    )
)

# Model data
model = GaussianNaiveBayes(dp.dataFrame)

# Get prediction from training and test sets
model.generatePrediction(
    model.trainingData[["Kontraktion", "Hyperæmi", "Ødem", "Eksudat"]],
    model.trainingData["Infektionsniveau"],
    model.testData[["Kontraktion", "Hyperæmi", "Ødem", "Eksudat"]],
    model.testData["Infektionsniveau"],
)


accplt = AccuracyPlotter()
xp, yp = model.getResults()
accplt.barPlot("Accuracy", xp, yp)
