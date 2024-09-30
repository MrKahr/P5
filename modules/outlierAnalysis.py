from dataPreprocessing import DataProcessor, Dataset
import matplotlib.pyplot as plt
import math


def AVF(row):
    sum = 0
    keys = row.keys()
    i = 0
    for attributeVal in row:
        sum += frequency(attributeVal, keys[i])
        i += 1
    return (1/len(row))*sum
    

def frequency(value, attribute, frequencies={}):
    # frequencies is only initialized once
    if not frequencies.get(attribute):
        frequencies[attribute] = ratio(attribute)
    return frequencies[attribute][value]
    


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


# Process data
dp = DataProcessor(Dataset.REGS)
df = dp.getDataFrame()

df.drop(["Gris ID", "Sår ID"], axis = 1, inplace = True)

listAVF = []
sum = 0
for index, row in df.iterrows():
    AVFelem = AVF(row)
    listAVF.append(AVFelem)
    sum += AVFelem

avg = sum / len(listAVF)
lowestPercentage = math.floor(len(listAVF) * 0.01)
print("average:", avg)
sortedList = sorted(listAVF)[:lowestPercentage]
print("outliers:", sortedList)

n, bins, patches = plt.hist(listAVF, bins=40)
sumBars = 0
i = 0
while sumBars <= lowestPercentage:
    patches[i].set_color("r")
    sumBars += n[i]
    i += 1
plt.xlabel("AVF score")
plt.ylabel("Datapoints in range")
plt.suptitle("Lowest 1% of scores shown in red")
plt.show()