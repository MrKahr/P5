# Main entry-point into our code base
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataPreprocessing.preprocessor import DataPreprocessor, Dataset
from modules.dataPreprocessing.transformer import DataTransformer

dp = DataPreprocessor(Dataset.REGS)
cleaner = DataCleaner(dp.df)

cleaner.cleanRegsDataset()
transformer = DataTransformer(cleaner.getDataframe())
# cleaner.deleteMissingValues()
transformer.KNNImputation()
