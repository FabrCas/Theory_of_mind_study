import numpy as np
import os
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

pathToDataset = './dataset/emotionSensor'
nameDataset = 'Andbrain_DataSet.csv'

for dirname, _, filenames in os.walk(pathToDataset):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv(pathToDataset + '/' + nameDataset)


# general info of the .csv file
print(" ******************** info ******************** \n")
data.info()

# label for classes
print("******************** labels ******************** \n")
print(data.columns)
#print(type(data.columns))

# correlation between classes
print(" ***************** correlation **************** \n")
print(data.corr())

# graphic correlation 
f,ax = plt.subplots(figsize=(30 ,30))
sns.heatmap(data.corr(),annot = True, linewidths = 0.6, fmt = ".4f", ax=ax)
plt.show()
# show the first ten rows

print(" ***************** 0-10 rows ***************** \n")
print(data[:10])  #data.head(10)
