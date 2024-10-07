# Code Style for project
# Setup:
# 1. Venv
# 2. Black Formatter
# 3. Comment Anchors
# 4. Autodocstring
# 5. Python Language Pack (Pylance, Python, Debugger)
"""
Casing of class, function, variable etc
 - Classes should be written as PascalCase
 - Functions should be written as camelCase
 - Variables should be written as snake_case
 - Global variables and constants should be written as ALLCAPS

 Casing of classes and functions as variables take precedence over the default variable casing
 Example:
"""
class TestClass():
    pass

def testFunction(TestClass: TestClass):
    a = TestClass()

testVar = testFunction
testVar()


"""
Functions should be type-annotated and have appropriate documentation
(Hint: You can create a docstring template by typing triple " + enter at the beginning of a function or class)
Example:
"""
def oneHotEncoding(self, labels: list[str]) -> None:
    """One-hot encode one or more categorical attributes

    Based on https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python

    Parameters
    ----------
    labels : list[str]
        A list of the column labels to one-hot encode
    """
    for variable in labels:
        one_hot = variable


"""
Private methods in classes should be prefixed with underscore, i.e., _
and be placed at the top of the class methods (but below methods with double underscore, i.e., __)

Note: a single _ does not make the class method private, but tells other devs they should treat it as such.

Note2: double __ does make the class method kinda private by prepending the class name to
       the method in the attribute table (I think its called that).
       This means a child class cannot directly access a __ method of a parent class
Example:
"""
class Params:
    def __init__(self, param: int) -> None: # This is private method, listed at the top
        self.param = param

    def _formatParam(self) -> int: # This is a "intended" private method, listed below __init__
        return self.param

    def setParams(self) -> int: ... # This is a public method, listed below _formatParam


"""
Use the anchors provide by the VSCode extension "Comment Anchors" to highlight things in the code, e.g.,
todos, issues, notes, links to other files etc.
Example:
"""
# TODO: This should todos
#NOTE - Note this
#FIXME - This is a bug
#LINK: modules\dataPreprocessing\preprocessor.py