# Main entry-point into our code base
from modules.dataPreprocessing.preprocessor import DataProcessor, Dataset
from modules.modelSelection.model_selection import GaussianNaiveBayes
from modules.dataExploration.visualization import Plotter

# Process data
dp = DataProcessor(Dataset.REGS)

# Model data
model = GaussianNaiveBayes(dp.df)

# Get prediction from training and test sets
model.generatePrediction(
    model.trainingData[["Kontraktion", "Hyperæmi", "Ødem", "Eksudat"]],
    model.trainingData["Infektionsniveau"],
    model.testData[["Kontraktion", "Hyperæmi", "Ødem", "Eksudat"]],
    model.testData["Infektionsniveau"],
)

accplt = Plotter()
xp, yp = model.getResults()
accplt.barPlot("Accuracy", xp, yp)
