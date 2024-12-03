from copy import deepcopy
import numpy as np
import math
from itertools import product
from scipy.stats._discrete_distns import randint
from scipy.stats._continuous_distns import uniform_gen

from modules.config.config import Config
from modules.config.grid_config import GridConfig
from modules.config.utils.config_enums import Model, VariableDistribution
from modules.logging import logger


class ParamGridGenerator:
    _logger = logger

    def __init__(self, feature_count: int) -> None:
        """
        Creates paramgrids for use in GridSearch/RandomSearch and friends.

        Parameters
        ----------
        feature_count : int
            Number of features seen during fit.
            Used solely for determining the size of the input layer to a MLPClassifier.
        """
        self._config = Config()
        self._grid_config = GridConfig()
        self._feature_count = feature_count

    def _getRange(
        self, start: int | float, stop: int | float, step: int | float
    ) -> list[int | float]:
        """
        Create a range of discrete or continous values, depending on input type.

        Parameters
        ----------
        start : int | float
            Beginning of sample space.

        stop : int | float
            Ending of sample space.
            NOTE: Stop is inclusive.

        step : int | float
            Samples generated in `step` steps.

        Returns
        -------
        list[int | float]
            If any parameter is a float, return a continous range. Otherwise, return discrete.
        """
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

    def _createNNTuple(self, key_hidden_layer_sizes: dict) -> list[tuple]:
        """
        Create combinations of tuples defining Neural Network layers and neurons per layer.

        Parameters
        ----------
        hidden_layer_sizes : dict
            The key `hidden_layer_sizes` as defined in the config.

        Returns
        -------
        list[tuple]
            A list of tuples defining different combinations of
            layers/neurons for training different sized Neural Networks in Grid-/Random Search.
        """
        # How many hidden layers we want  to use for each model
        layer_range = self._getRange(**key_hidden_layer_sizes["layers"])
        low, high, size = key_hidden_layer_sizes["layer_size"].values()

        # Get the size of the input layer
        input_layer = self._getRange(
            start=self._feature_count, stop=self._feature_count, step=1
        )

        # Each element represents the size of a hidden layer
        hidden_layer_sizes = [
            randint.rvs(low, high, size=i).astype(dtype="int32") for i in layer_range
        ]  # type: list[np.ndarray]

        # Get the size of the output layer
        output_layer = self._getRange(**key_hidden_layer_sizes["output_layer"])

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
        """
        Creates a generic grid for use with RandomSearch.

        Parameters
        ----------
        value : dict
            The grid as defined in the config.

        Returns
        -------
        np.ndarray | None
            The grid with distributions as required by Random Search.
        """
        distribution = None
        if isinstance(value, dict) and "dist" in value:
            distribution_type = value["dist"]
            params = deepcopy(value["dist_params"])  # type: dict
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
        """
        Creates a generic grid for use with GridSearch.

        Parameters
        ----------
        grid : dict
            The grid as defined in the config.

        Returns
        -------
        dict
            The grid with ranges as required by GridSearch.
        """
        new_grid = {}
        for k, v in grid.items():
            if k == "hidden_layer_sizes":
                new_grid |= {k: list(self._createGenericGrid(v).values())}
            elif isinstance(v, dict):
                new_grid |= {k: self._getRange(**v)}
            else:
                new_grid |= {k: v}
        return new_grid

    def _createGenericRandomGrid(self, grid: dict) -> dict:
        """
        Creates a grid of parameter distributions for use with Random Search.

        Parameters
        ----------
        grid : dict
            The grid as defined in the config.

        Returns
        -------
        dict
            The grid with distributions as required by Random Search.
        """
        distribution = {}
        for k, v in grid.items():
            distribution |= {k: self._createGenericDistribution(v)}
        return distribution

    def _createNNRandomGrid(self, grid: dict) -> dict:
        """
        Creates a grid of distributions for Neural Network parameters
        for use with RandomSearch.

        Parameters
        ----------
        grid : dict
            The Neural Network grid as defined in the config.

        Returns
        -------
        dict
            The Neural Network grid with distributions as required by Random Search.
        """
        # Hidden layer sizes must be handled seperately from other key value pairs because they are used to generate tuples
        distribution = {}
        for k, v in grid.items():
            if k == "hidden_layer_sizes":
                distribution |= {k: self._createNNTuple(v)}
            else:
                distribution |= {k: self._createGenericDistribution(v)}
        return distribution

    def getParamGrid(self) -> dict:
        """
        Create a parameter grid of values for use with GridSearchCV or similar.

        Returns
        -------
        dict
            The parameter grid as defined in the config.
        """
        # Initialize model and grid
        current_model = self._config.getValue("model", "ModelSelection")

        # Ensure param grid matches model
        match current_model:
            case Model.DECISION_TREE.name:
                grid_key = "ParamGridDecisionTree"
                grid = self._createGenericGrid(self._grid_config.getValue(grid_key))
            case Model.NAIVE_BAYES.name:
                grid_key = "RandomParamGridGaussianNaiveBayes"
                grid = self._createGenericGrid(self._grid_config.getValue(grid_key))
            case Model.NEURAL_NETWORK.name:
                grid_key = "ParamGridNeuralNetwork"
                grid = self._createGenericGrid(self._grid_config.getValue(grid_key))
            case Model.RANDOM_FOREST.name:
                tree_key = "ParamGridDecisionTree"
                grid_key = "ParamGridRandomForest"
                grid = self._createGenericGrid(self._grid_config.getValue(tree_key))
                grid |= self._createGenericGrid(self._grid_config.getValue(grid_key))
            case _:
                self._logger.error(
                    f"Paramgrid not supported for model '{current_model}'. Expected one of '{Model._member_names_}'"
                )
        return grid

    def getRandomParamGrid(self) -> dict:
        """
        Create a parameter grid of distributions for use with RandomizedSearchCV or similar.

        Returns
        -------
        dict
            The parameter grid of distributions as defined in the config.
        """
        # Initialize model and grid
        current_model = self._config.getValue("model", "ModelSelection")

        # Ensure param grid matches model
        match current_model:
            case Model.DECISION_TREE.name:
                grid_key = "RandomParamGridDecisionTree"
                grid = self._createGenericRandomGrid(
                    self._grid_config.getValue(grid_key)
                )
            case Model.NAIVE_BAYES.name:
                grid_key = "RandomParamGridGaussianNaiveBayes"
                grid = self._createGenericRandomGrid(
                    self._grid_config.getValue(grid_key)
                )
            case Model.NEURAL_NETWORK.name:
                grid_key = "RandomParamGridNeuralNetwork"
                grid = self._createNNRandomGrid(self._grid_config.getValue(grid_key))
            case Model.RANDOM_FOREST.name:
                # We need to combine keys since random forest uses decision tree params to get keys
                tree_key = "RandomParamGridDecisionTree"
                grid_key = "RandomParamGridRandomForest"
                grid = self._createGenericRandomGrid(
                    self._grid_config.getValue(tree_key)
                )
                grid |= self._createGenericRandomGrid(
                    self._grid_config.getValue(grid_key)
                )
            case _:
                self._logger.error(
                    f"Paramgrid not supported for model '{current_model}'. Expected one of '{Model._member_names_}'"
                )
        return grid
