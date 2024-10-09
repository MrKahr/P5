# Main entry-point into our code base
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataPreprocessing.preprocessor import DataPreprocessor, Dataset


dp = DataPreprocessor(Dataset.OLD)
cleaner = DataCleaner(dp.df)
cleaner.cleanOldDataset()
