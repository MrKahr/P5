from dataPreprocessing import DataProcessor, Dataset
import pandas as pd



def AVF(row, df):
    # attributes = list(elem)
    # print(type(row))
    sum = 0
    keys = row.keys()
    i = 0
    for attributeVal in row:
        # print(attribute)
        # print("\n\n")
        # print(df[0][0])
        sum += frequency(attributeVal, keys[i])
        i += 1
    return (1/len(row))*sum
    

def frequency(value, attribute):
    return ratio(attribute)[value]
    

# for elem in df.iterrows():
#     AVF(elem)

# freqDict = dict()
# columns = df.columns
# for colName in columns:
#     freqDict.update(colName)
#     #print(df[colName])

def ratio(attribute: str) -> dict:
    column = df[attribute]
    values = column.unique()
    sums = {}
    for i in values:
        sums[i] = 0
    for i in column:
        sums[i] += 1
    for i in values:
        sums[i] = sums[i] / len(column)
    return sums

# print(df[0][0])

# Process data
dp = DataProcessor(Dataset.REGS)
df = dp.dataFrame

listAVF = []
sum = 0
for index, row in df.iterrows():
    AVFelem = AVF(row, df)
    listAVF.append(AVFelem)
    sum += AVFelem

avg = sum / len(listAVF)
print(avg)

