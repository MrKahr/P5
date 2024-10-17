from modelSelection.model_name_enum import ModelName
from modelSelection.model_selection import model_selection
from modelTraining.model_training import model_training
from modelTraining.scoring_function import scoring_function
from modelTraining.scoring_function_enum import ScoringFunctionName
from dataPreprocessing.preprocessor import DataPreprocessor
from dataPreprocessing.dataset_enums import Dataset


def pipeline():
    def testing(estimator, X, y):
        print(
            f"Unseen test data accuracy: {scoring_function(estimator, X, y, max_prediction=TEST_Y_MAX, sfname=ScoringFunctionName.THRESHOLDACCURACY):.3f}"
        )

    # Unprocessed data
    def load_training_data(type: Dataset = Dataset.REGS):
        global TRAIN_Y_MAX
        print("Loading training data")
        dp = DataPreprocessor(type)
        TRAIN_Y_MAX = dp.getTargetMaxValue()
        return (
            dp.getTrainingData(),
            dp.getTargetData(),
            dp.getTrainingLabels(),
        )

    def load_testing_data(type: Dataset = Dataset.OLD):
        global TEST_Y_MAX
        print("Loading testing data")
        dp = DataPreprocessor(type)
        TEST_Y_MAX = dp.getTargetMaxValue()
        return (
            dp.getTrainingData(),
            dp.getTargetData(),
            dp.getTrainingLabels(),
        )

    # Untrained model
    untrainedModel = model_selection(ModelName.DECISIONTREE)
    X_train, Y_train, train_labels = load_training_data(Dataset.REGS)
    X_test, Y_test, test_labels = load_testing_data(Dataset.OLD)
    testingData = load_training_data()
    # Trained model
    predictor = model_training(
        ModelName.DECISIONTREE,
        untrainedModel,
        "fit",
        scoring_function(sfname=ScoringFunctionName.THRESHOLDACCURACY),
        X_train,
        Y_train,
    )

    testing(predictor, X_test, Y_test)
