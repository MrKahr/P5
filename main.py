# Main entry-point into our code base
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataPreprocessing.preprocessor import DataPreprocessor, Dataset


dp = DataPreprocessor(Dataset.REGS)
cleaner = DataCleaner(dp.df)
cleaner.cleanRegsDataset()
cleaner.deleteMissingValue()
cleaner.showRowRemovalRatio()
