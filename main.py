# Main entry-point into our code base
from modules.dataPreprocessing import DataProcessor, Dataset
from modules.models import GaussianNaiveBayes
from modules.visualization import AccuracyPlotter

# Process data
dp = DataProcessor(Dataset.MÅL)
dp.showDataFrame()
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
