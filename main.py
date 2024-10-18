##########################
### Initial Path Setup ###
##########################
# Set initial CWD
import os

from modules.dataPreprocessing.processor import Processor
from modules.dataPreprocessing.strategy_parse_config import StrategyparseConfig
from modules.modelSelection.model_selection import ModelSelector
from modules.modelTraining.model_trainer import ModelTrainer
from modules.modelTraining.strategy_fit_decision_tree import StrategyFitDecisionTree

os.chdir(os.path.dirname(os.path.abspath(__file__)))

######################
### Module Imports ###
######################

from modules.config.config import Config
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataPreprocessing.feature_selector import FeatureSelector
from modules.dataPreprocessing.preprocessor import DataPreprocessor, Dataset
from modules.dataPreprocessing.transformer import DataTransformer

config = Config()
dataProcessor = DataPreprocessor(Dataset.REGS)
processor = Processor(
    config.getValue("Cleaning"),
    PipelineComponent=DataCleaner(dataProcessor.df),
    Strategy=StrategyparseConfig(),
)
processor.performAlgorithm()


def custom_score(estimator, X, y):
    def threshold(result, y, th: int = 5):
        score = 0
        for i, prediction in enumerate(result):
            if y[i] >= th:
                score += 1
            elif prediction < th:
                score += 1
        return score

    result = estimator.predict(X)
    score = threshold(result, y)
    # score = threshold(result, y, th=20)
    return float(score / len(result))


selector = ModelSelector
estimator = selector.getModel()
DataCleaner(dataProcessor.df).cleanRegsDataset()
features = dataProcessor.getTrainingData()
target = dataProcessor.getTargetData()
trainer = ModelTrainer(estimator, features, target, StrategyFitDecisionTree)
fittedModel = trainer.fitModel()
n_features = fittedModel.n_features_in_

print(f"Optimal features: {n_features}")
print(f"Training data accuracy: {custom_score(fittedModel, features, target):.3f}")
