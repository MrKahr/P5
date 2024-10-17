# cross_vaidator_selection method takes CrossValidatorName enum as option, returns CrossValidator of the chosen type.


from modules.modelTraining.scoring_function_enum import ScoringFunctionName


def threshold(result, y, th: int = 5):
    score = 0
    for i, prediction in enumerate(result):
        if y[i] >= th:
            score += 1
        elif prediction < th:
            score += 1
    return score


def distance(result, y, max_prediction=0):
    score = 0
    for i, prediction in enumerate(result):
        score += 1 - abs(prediction - y[i]) / max_prediction
    return score


def scoring_function(
    estimator,
    X,
    y,
    max_prediction=None,
    sfname: ScoringFunctionName = ScoringFunctionName.THRESHOLDACCURACY,
):

    match sfname:
        case ScoringFunctionName.THRESHOLDACCURACY:
            result = estimator.predict(X)
            score = threshold(result, y, max_prediction)
            # score = threshold(result, y, th=20)
            return float(score / len(result))
        case ScoringFunctionName.DISTANCEACCURACY:
            result = estimator.predict(X)
            score = distance(result, y, max_prediction)
            # score = threshold(result, y, th=20)
            return float(score / len(result))
