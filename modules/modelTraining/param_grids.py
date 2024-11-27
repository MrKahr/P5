from itertools import product
import math
from scipy.stats._discrete_distns import randint
from scipy.stats._continuous_distns import uniform_gen
import numpy as np
from modules.config.config import Config
from modules.config.config_enums import Model, VariableDistribution


class ParamGridGenerator:
    def __init__(self, feature_count: int) -> None:
        self._config = Config()
        self._feature_count = feature_count

    def _getRange(
        self, start: int | float, stop: int | float, step: int | float
    ) -> list[int | float]:
        # In range(), "stop" is exclusive. Ranges for an element of size 1 is modified to include the element itself.
        stop += 1
        if type(start) is float or type(stop) is float or type(step) is float:
            # linspace requires number of steps and not stepsize
            value_range = list(
                np.linspace(start, stop, math.ceil(abs(stop - start) / step))
            )
        else:
            value_range = list(range(start, stop, step))
        return value_range

    def getParamGrid(self) -> dict:
        # TODO: Implement more input validation
        # Initialize model and grid
        current_model = self._config.getValue("model", "ModelSelection")

        # Ensure param grid matches model
        match current_model:
            case Model.DECISION_TREE.name:
                grid_key = "ParamGridDecisionTree"
            case Model.NAIVE_BAYES.name:
                grid_key = "RandomParamGridGaussianNaiveBayes"
            case Model.NEURAL_NETWORK.name:
                grid_key = "ParamGridNeuralNetwork"
            case Model.RANDOM_FOREST.name:
                grid_key = "ParamGridRandomForest"

        current_grid = self._config.getValue(grid_key)

        # Update current config
        for k, v in current_grid.items():
            if isinstance(v, dict):
                self._config.setValue(k, self._getRange(**v), grid_key)

        # Param grid updated with ranges
        current_grid = self._config.getValue(grid_key)
        return current_grid

    def _createGenericDistribution(self, grid: dict) -> dict:
        for k, v in grid.items():
            if isinstance(v, dict):
                if "dist" in v:
                    print(v)
                    distribution_type = v["dist"]
                    params = v["dist_params"]  # type: dict
                    if distribution_type == VariableDistribution.RANDINT.name:
                        distribution = randint.rvs(**v["dist_params"]).astype(
                            dtype="int32"
                        )
                    elif distribution_type == VariableDistribution.RANDFLOAT.name:
                        distribution = uniform_gen(
                            a=params.pop("low"),
                            b=params.pop("high"),
                            name="uniform2",
                        ).rvs(**params)
                    print(distribution)
        return distribution

    # TODO: handle NN edge cases
    def _createNNDistributions(self, grid: dict) -> dict:
        for k, v in grid.items():
            if "hidden_layer_sizes" == k:
                layer_range = self._getRange(**v["layers"])
                layer_size_range = self._getRange(**v["layer_size"])

                layer_size_list = []  # type: list[tuple]
                print(layer_range)
                for i in layer_range:
                    print(i)
                    layer_size_list.append(list(product(layer_size_range, repeat=i)))
                print(layer_size_list)

                input_layer_range = self._getRange(
                    start=self._feature_count, stop=self._feature_count, step=1
                )
                output_layer_range = self._getRange(**v["output_layer"])
                layer_tuples = product(
                    input_layer_range, layer_range, layer_size_range, output_layer_range
                )
                # print(list(layer_tuples))

            if isinstance(v, dict):
                if "dist" in v:
                    # print(v)
                    distribution_type = v["dist"]
                    params = v["dist_params"]  # type: dict
                    if distribution_type == VariableDistribution.RANDINT.name:
                        distribution = randint.rvs(**v["dist_params"]).astype(
                            dtype="int32"
                        )
                    elif distribution_type == VariableDistribution.RANDFLOAT.name:
                        distribution = uniform_gen(
                            a=params.pop("low"),
                            b=params.pop("high"),
                            name="uniform2",
                        ).rvs(**params)
                    # print(distribution)
        return distribution

    def getRandomParamGrid(self) -> dict:
        # Initialize model and grid
        parent_key = "RandomParamGrid"
        current_model = self._config.getValue("model", "ModelSelection")

        # Ensure param grid matches model
        match current_model:
            case Model.DECISION_TREE.name:
                grid_key = "RandomParamGridDecisionTree"
            case Model.NAIVE_BAYES.name:
                grid_key = "RandomParamGridGaussianNaiveBayes"
            case Model.NEURAL_NETWORK.name:
                grid_key = "RandomParamGridNeuralNetwork"
                self._config.setValue(
                    grid_key,
                    self._createNNDistributions(
                        self._config.getValue(grid_key, parent_key)
                    ),
                    parent_key,
                )
            case Model.RANDOM_FOREST.name:
                grid_key = "RandomParamGridRandomForest"

        return self._config.getValue(grid_key, parent_key)
