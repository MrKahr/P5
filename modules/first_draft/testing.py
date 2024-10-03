import sys
import os
from time import time

from matplotlib import pyplot as plt

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, plot_tree

sys.path.insert(0, os.getcwd())
plt.ion()


from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from modules.dataPreprocessing.preprocessor import DataProcessor, Dataset


# NOTE: The random state should be set to a constant integer to get reproducible results across iterations
#       However, be aware of: https://scikit-learn.org/stable/common_pitfalls.html#robustness-of-cross-validation-results

# NOTE: If using permutation importance and it shows that no features are important, consider:
#       https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py

TRAIN_Y_MAX = 0
TEST_Y_MAX = 0


def load_training_data(type: Dataset = Dataset.REGS):
    global TRAIN_Y_MAX
    print("Loading training data")
    dp = DataProcessor(type)
    TRAIN_Y_MAX = dp.getTargetMaxValue()
    return (
        dp.getTrainingData(),
        dp.getTargetData(),
        dp.getTrainingLabels(),
    )


def load_testing_data(type: Dataset = Dataset.OLD):
    global TEST_Y_MAX
    print("Loading testing data")
    dp = DataProcessor(type)
    TEST_Y_MAX = dp.getTargetMaxValue()
    return (
        dp.getTrainingData(),
        dp.getTargetData(),
        dp.getTrainingLabels(),
    )


# SECTION: Dimensionality reduction (removing unpredictive features)
# TODO: Implement method for visual representation as well

# Conditional Indepence testing. Start with simple Chi-2 and try to implement CPI paper
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.chi2_contingency.html


# Feature selection
# https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection


# Feature selection
# https://scikit-learn.org/stable/api/sklearn.feature_selection.html


# SECTION: Models
# The model to fit (a.k.a. an estimator)
def the_model(m: str):
    print("Loading model: ", end="")
    if m == "dtc":
        print("DecisionTreeClassifier")
        estimator = DecisionTreeClassifier(
            criterion="entropy", max_depth=None, random_state=1
        )
    elif m == "rfc":
        print("RandomForestClassifier")
        estimator = RandomForestClassifier(
            criterion="entropy", max_depth=None, random_state=0
        )
    elif m == "nb":
        print("GaussianNB")
        estimator = GaussianNB()
    return estimator


# SECTION: Cross-Validator
# Time Series: https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-of-time-series-data
# Stratification: https://scikit-learn.org/stable/modules/cross_validation.html#stratified-k-fold
def cross_validator(t: str):
    print("Using cross-validator: ", end="")
    cv = None
    if t == "ts":
        print("TimeSeriesSplit")
        cv = TimeSeriesSplit(n_splits=5)
    elif t == "skf":
        print("StratifiedKFold")
        cv = StratifiedKFold(n_splits=5)
    return cv


# SECTION: Tuning Score
def custom_score(estimator, X, y, max_prediction=None):
    def threshold(result, y, th: int = 5):
        score = 0
        for i, prediction in enumerate(result):
            if y[i] >= th:
                score += 1
            elif prediction < th:
                score += 1
        return score

    def distance(result, y, max_prediction):
        if max_prediction is None:
            max_prediction = TRAIN_Y_MAX
        score = 0
        for i, prediction in enumerate(result):
            score += 1 - abs(prediction - y[i]) / max_prediction
        return score

    result = estimator.predict(X)
    score = distance(result, y, max_prediction)
    # score = threshold(result, y, th=20)
    return float(score / len(result))


# SECTION: Tuning hyperparameters of an estimator (e.g. optimal CV settings)
# TODO: Implement method for visual representation as well
# The big idea:
#   Cross validation iterators can also be used to directly perform model
#   selection using Grid Search for the optimal hyperparameters of the model.
# Guide: https://scikit-learn.org/stable/modules/grid_search.html
# Feature selection methods:
# Recursive Feature Elimination with Cross-Validation (RFECV)
# NOTE: This is also automatic feature selection
# https://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html
# Brute-force search:
# Check guide
# Random search:
# Check guide
def training(estimator: DecisionTreeClassifier, X, Y, cv, tuning: bool):
    start_time = time()
    print(f"Training model {"with tuning" if tuning else "without tuning"}...")
    if tuning:
        tuner = RFECV(estimator=estimator, cv=cv, scoring=custom_score, n_jobs=-1)
        tuner.fit(X, Y)
        end_time = time()
        n_features = tuner.n_features_
        fitted = tuner

        cv_results = pd.DataFrame(tuner.cv_results_)
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Mean test accuracy")
        plt.errorbar(
            x=cv_results["n_features"],
            y=cv_results["mean_test_score"],
            yerr=cv_results["std_test_score"],
        )
        plt.title("Recursive Feature Elimination with Cross-Validation\n(with correlated features)")
        plt.show()
    else:
        fitted = estimator.fit(X, Y)
        end_time = time()
        n_features = fitted.n_features_in_
    print(f"Training completed in {end_time-start_time:.3f} s")
    print(f"Optimal features: {n_features}")
    print(f"Training data accuracy: {custom_score(fitted, X, Y):.3f}")
    return fitted


# SECTION: Feature importance
# TODO: Implement method for visual representation as well
# Example: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
# NOTE: Features that are deemed of low importance for a bad model (low cross-validation score)
#       could be very important for a good model. Therefore it is always important to evaluate
#       the predictive power of a model using a held-out set (or better with cross-validation) prior to computing importances.
#       Permutation importance does not reflect the intrinsic predictive value of a feature by itself but how important this
#       feature is for a particular model.
def feature_importance(estimator, X, Y, feature_names):
    print(f"Calculating feature importance")
    result = permutation_importance(
        estimator, X, Y, n_repeats=10, random_state=42, n_jobs=-1
    )
    feature_importances = pd.Series(result.importances_mean, index=feature_names)

    fig, ax = plt.subplots()
    feature_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title(
        "Feature importances using permutation on full model\n(this is model-specific)"
    )
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()


def testing(estimator, X, y):
    print(
        f"Unseen test data accuracy: {custom_score(estimator, X, y, max_prediction=TEST_Y_MAX):.3f}"
    )


# Model evaluation (for reference)
# https://scikit-learn.org/stable/modules/model_evaluation.html


if __name__ == "__main__":
    is_tuning = False
    X_train, Y_train, train_labels = load_training_data(Dataset.REGS)
    model = the_model("nb")
    cv = cross_validator("skf")
    fitted_model = training(model, X_train, Y_train, cv, tuning=is_tuning)
    feature_importance(fitted_model, X_train, Y_train, train_labels)
    X_test, Y_test, test_labels = load_testing_data(Dataset.OLD)
    testing(fitted_model, X_test, Y_test)
    input("Press enter to exit")
