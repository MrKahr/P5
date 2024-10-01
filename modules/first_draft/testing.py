import sys
import os

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.getcwd())

from sklearn.feature_selection import RFECV
from sklearn.model_selection import TimeSeriesSplit
from modules.dataPreprocessing import DataProcessor, Dataset


# NOTE: The random state should be set to a constant integer to get reproducible results across iterations
#       However, be aware of: https://scikit-learn.org/stable/common_pitfalls.html#robustness-of-cross-validation-results

# NOTE: If using permutation importance and it shows that no features are important, consider:
#       https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py


dp = DataProcessor(type=Dataset.REGS)
df = dp.getDataFrame()

# Dimensionality reduction (removing unpredictive features)
# TODO: Implement method for visual representation as well
# Conditional Indepence testing. Start with simple Chi-2 and try to implement CPI paper


# The model to fit (a.k.a. an estimator)
estimator = DecisionTreeClassifier(criterion="entropy", max_depth=None, random_state=1)


# Cross-Validator
# Time Series: https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-of-time-series-data
# Stratification: https://scikit-learn.org/stable/modules/cross_validation.html#stratified-k-fold
cv = TimeSeriesSplit(n_splits=5)


# Tuning hyperparameters of an estimator (e.g. optimal CV settings)
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
tuner = RFECV(estimator=estimator, cv=cv, scoring="accuracy", n_jobs=-1)
tuner.fit(X=dp.getTrainingData(), y=dp.getTargetData())


print(f"Optimal features: {tuner.n_features_}")


cv_results = pd.DataFrame(tuner.cv_results_)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Mean test accuracy")
plt.errorbar(
    x=cv_results["n_features"],
    y=cv_results["mean_test_score"],
    yerr=cv_results["std_test_score"],
)
plt.title("Recursive Feature Elimination \nwith correlated features")
plt.show()


# Feature selection
# https://scikit-learn.org/stable/api/sklearn.feature_selection.html


# Feature importance
# TODO: Implement method for visual representation as well
# Example: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html


# Model evaluation (for reference)
# https://scikit-learn.org/stable/modules/model_evaluation.html
