# Modelselection method takes model name enum as option, returns unfitted model of the chosen type.
from model_name_enum import ModelName
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def model_selection(mname: ModelName):
    match mname:
        case ModelName.DECISIONTREE:
            print("DecisionTreeClassifier")
            return DecisionTreeClassifier(
                criterion="entropy", max_depth=None, random_state=1
            )
        case ModelName.RANDOMFOREST:
            return RandomForestClassifier(
                criterion="entropy", max_depth=None, random_state=0
            )
        case ModelName.SUPPORTVECTOR:
            return SVC()
        case ModelName.NAIVEBAYES:
            return GaussianNB()
