import sys
import os

sys.path.insert(0, os.getcwd())

from modules.dataPreprocessing.outlier_analysis.outlier_processor import OutlierProcessor
from modules.dataPreprocessing.preprocessor import DataPreprocessor, Dataset
from modules.dataPreprocessing.cleaner import DataCleaner
from modules.dataPreprocessing.transformer import DataTransformer

if __name__ == "__main__":
    dp = DataPreprocessor(Dataset.REGS)

    cleaner = DataCleaner(dp.df)
    cleaner.cleanRegsDataset()
    cleaner.deleteMissingValues()

    transformer = DataTransformer(cleaner.getDataframe())
    transformer.oneHotEncode(["Eksudattype", "Hyperæmi"])

    outlier_remover = OutlierProcessor(transformer.getDataframe())


    # outlier_remover.df.drop(["Gris ID", "Sår ID"], axis=1, inplace=True) # Outlier removal will remove these in a shallow copy, but still works if they have been removed beforehand

    # Use to update these numbers in report, in case of changes
    # print(len(outlier_remover.odin(10, 0)))
    # print(len(outlier_remover.odin(20, 0)))
    # print(len(outlier_remover.odin(30, 0)))

    # Use to check different/combined outlier removal methods
    outlier_remover.removeOutliers(outliers=outlier_remover.avf(10))
    outlier_remover.removeOutliers(outliers=outlier_remover.odin(30, 0))

    # Use to check DataFrame after removing outliers
    # print(outlier_remover.df)