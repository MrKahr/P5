from itertools import product
import math
from scipy.stats._discrete_distns import randint
from scipy.stats._continuous_distns import uniform_gen
import numpy as np
from modules.config.config import Config
from modules.config.config_enums import Model, VariableDistribution
from modules.logging import logger


class ParamGridGenerator:
    _logger = logger

    def __init__(self, feature_count: int) -> None:
        self._config = Config()
        self._feature_count = feature_count

    def _getRange(
        self, start: int | float, stop: int | float, step: int | float
    ) -> list[int | float]:
        if type(start) is float or type(stop) is float or type(step) is float:
            # linspace requires number of steps and not stepsize
            value_range = list(
                np.linspace(start, stop, math.ceil(abs(stop - start) / step))
            )
        else:
            # In range(), "stop" is exclusive. Ranges for an element of size 1 is modified to include the element itself.
            stop += 1
            value_range = list(range(start, stop, step))

        return value_range

    def _createNNTuple(self, v: dict) -> list[tuple]:
        # How many hidden layers we want  to use for each model
        layer_range = self._getRange(**v["layers"])
        low, high, size = v["layer_size"].values()

        # Get the size of the input layer
        input_layer = self._getRange(
            start=self._feature_count, stop=self._feature_count, step=1
        )

        # Each element represents the size of a hidden layer
        hidden_layer_sizes = [
            randint.rvs(low, high, size=i).astype(dtype="int32") for i in layer_range
        ]  # type: list[np.ndarray]

        # Get the size of the output layer
        output_layer = self._getRange(**v["output_layer"])

        # Combine lists in tuples for different Neural Network (NN) sizes since MLP-constructor requires tuple input
        # It is done in format (INPUT,a,b,...,OUTPUT) e.g. (10, 5, 8, 2)
        # Input: 10 neurons, Layer1: 5 neurons, Layer2: 8 neurons, Output: 2 neurons.
        # To create tuple, hidden_layer_sizes, are arrays, not literals.
        # There are unpacked in the final NN tuple
        layer_tuples = []
        # Cartesian product of all layer types
        for layer_tuple in product(input_layer, hidden_layer_sizes, output_layer):
            # Container for tuples
            temp = []
            for elem in layer_tuple:
                if isinstance(elem, (np.ndarray, list)):
                    temp.extend(elem)  # Unpack array in final NN tuple
                else:
                    temp.append(elem)  # Element is already a number. Add as is
            # Create the final NN tuple and add to list of NN tuples
            layer_tuples.append((tuple(temp)))
        return layer_tuples

    def _createGenericDistribution(self, value: dict) -> np.ndarray | None:
        distribution = None
        if isinstance(value, dict) and "dist" in value:
            distribution_type = value["dist"]
            params = value["dist_params"]  # type: dict
            if distribution_type == VariableDistribution.RANDINT.name:
                distribution = randint.rvs(**value["dist_params"]).astype(dtype="int32")
            elif distribution_type == VariableDistribution.RANDFLOAT.name:
                distribution = uniform_gen(
                    a=params.pop("low"),
                    b=params.pop("high"),
                    name="uniform2",
                ).rvs(**params)

        return distribution if distribution is not None else value

    def _createGenericGrid(self, grid: dict) -> dict:
        new_grid = {}
        for k, v in grid.items():
            if isinstance(v, dict):
                new_grid |= {k: self._getRange(**v)}
            else:
                new_grid |= {k: v}
        return new_grid

    def _createGenericRandomGrid(self, grid: dict) -> dict:
        distribution = {}
        for k, v in grid.items():
            distribution |= {k: self._createGenericDistribution(v)}
        return distribution

    def _createNNGrid(self, grid: dict) -> dict:
        # Hidden layer sizes must be handled seperately from other key value pairs because they are used to generate tuples
        distribution = {}
        for k, v in grid.items():
            if k == "hidden_layer_sizes":
                distribution |= {k: self._createNNTuple(v)}
            else:
                distribution |= {k: self._createGenericDistribution(v)}
        return distribution

    def getParamGrid(self) -> dict:
        # TODO: Implement more input validation
        # Initialize model and grid
        current_model = self._config.getValue("model", "ModelSelection")

        # Ensure param grid matches model
        match current_model:
            case Model.DECISION_TREE.name:
                grid_key = "ParamGridDecisionTree"
                grid = self._createGenericGrid(self._config.getValue(grid_key))
            case Model.NAIVE_BAYES.name:
                grid_key = "RandomParamGridGaussianNaiveBayes"
                grid = self._createGenericGrid(self._config.getValue(grid_key))
            case Model.NEURAL_NETWORK.name:
                grid_key = "ParamGridNeuralNetwork"
                grid = self._createGenericGrid(self._config.getValue(grid_key))
            case Model.RANDOM_FOREST.name:
                tree_key = "ParamGridDecisionTree"
                forest_key = "ParamGridRandomForest"
                grid = self._createGenericGrid(self._config.getValue(tree_key))
                grid |= self._createGenericGrid(self._config.getValue(forest_key))
            case _:
                self._logger.error(
                    f"Paramgrid not supported for model '{current_model}'. Expected one of '{Model._member_names_}'"
                )

        # Param grid updated with ranges
        self._config.setValue(grid_key, grid, "ParamGrid")
        print(grid)
        return grid

    def getRandomParamGrid(self) -> dict:
        # Initialize model and grid
        parent_key = "RandomParamGrid"
        current_model = self._config.getValue("model", "ModelSelection")

        # Ensure param grid matches model
        match current_model:
            case Model.DECISION_TREE.name:
                grid_key = "RandomParamGridDecisionTree"
                grid = self._createGenericRandomGrid(
                    self._config.getValue(grid_key, parent_key)
                )
            case Model.NAIVE_BAYES.name:
                grid_key = "RandomParamGridGaussianNaiveBayes"
                grid = self._createGenericRandomGrid(
                    self._config.getValue(grid_key, parent_key)
                )
            case Model.NEURAL_NETWORK.name:
                grid_key = "RandomParamGridNeuralNetwork"
                grid = self._createNNGrid(self._config.getValue(grid_key, parent_key))
            case Model.RANDOM_FOREST.name:
                # We need to combine keys since random forest uses decision tree params to get keys
                tree_key = "RandomParamGridDecisionTree"
                grid_key = "RandomParamGridRandomForest"
                grid = self._createGenericRandomGrid(
                    self._config.getValue(tree_key, parent_key)
                )
                grid |= self._createGenericRandomGrid(
                    self._config.getValue(grid_key, parent_key)
                )
            case _:
                self._logger.error(
                    f"Paramgrid not supported for model '{current_model}'. Expected one of '{Model._member_names_}'"
                )

        self._config.setValue(
            grid_key,
            grid,
            parent_key,
        )
        return self._config.getValue(grid_key, parent_key)
