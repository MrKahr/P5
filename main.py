# Main entry-point into our code base
from modules.dataPreprocessing import DataProcessor, Dataset
from modules.models import GaussianNaiveBayes
from modules.visualization import Plotter
from modules.continuousVariables import ContinuousPlotter

# Process data
dp = DataProcessor(Dataset.REGS)
condp = DataProcessor(Dataset.MÅL)
condp.showDataFrame()
# Model data
model = GaussianNaiveBayes(dp.dataFrame)

# Get prediction from training and test sets
model.generatePrediction(
    model.trainingData[["Kontraktion", "Hyperæmi", "Ødem", "Eksudat"]],
    model.trainingData["Infektionsniveau"],
    model.testData[["Kontraktion", "Hyperæmi", "Ødem", "Eksudat"]],
    model.testData["Infektionsniveau"],
)


accplt = Plotter()
xp, yp = model.getResults()
#accplt.barPlot("Accuracy", xp, yp)

conplt = ContinuousPlotter()

conplt.continousPlot(condp.dataFrame["Sårrand (cm)"], condp.dataFrame["Midte (cm)"], condp.dataFrame["Dag"])

