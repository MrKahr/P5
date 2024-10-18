##########################
### Initial Path Setup ###
##########################
# Set initial CWD
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

######################
### Module Imports ###
######################

from modules.selection.cross_validation_selection import CrossValidationSelector
from modules.selection.model_selection import ModelSelector

model = ModelSelector.getModel()
cv = CrossValidationSelector.getCrossValidator()
print(model)
print(cv)
