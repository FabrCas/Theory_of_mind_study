import numpy as np
import os
import time
import pandas as pd 
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split

pathToDataset = './dataset/emotionSensor'
pathToDatasetAbs = "/home/faber/Desktop/ProjectCode/Project: Theory of mind and emotional resonance study/dataset/emotionSensor"


nameDataset = 'Andbrain_DataSet.csv'
nameDataset_train = 'Andbrain_DataSet_train.csv'
nameDataset_test = 'Andbrain_DataSet_test.csv'

"""
target:
1) disgust
2) surprise
3) neutral
4) anger
5) sad
6) happy
7) fear
"""
class emotionSensorReader():
    def __init__(self):
       self.data = None
       self.trainSet = np.array([[]])   
       self.testSet = np.array([[]])
       
       self._computeDataFrame()
       
       
    def _computeDataFrame(self): 
        # for dirname, _, filenames in os.walk(pathToDataset):
            # for filename in filenames:
            #     print(os.path.join(dirname, filename))
        
        data = pd.read_csv(pathToDataset + '/' + nameDataset)
        self.data = data
    
    def getLabels(self, print_it = False):
        columns = np.array(self.data.columns.tolist())[1:]
        if print_it: print(columns)
        return columns
    
    def splitDataset(self, writeFiles = True, test_size = 0.2):
        n_instances = len(self.data)
        
        startTime = time.time()

        train, test = train_test_split(self.data, test_size= test_size, shuffle= True)
            
                
        self.trainSet = np.array(train)
        self.testSet = np.array(test)

        
        if(writeFiles):
            columns_names = self.getLabels()
            columns_names = np.insert(columns_names, 0,"word")
            # print(columns_names)
            

            dataframe_train = pd.DataFrame(self.trainSet)
            dataframe_test =  pd.DataFrame(self.testSet)
            dataframe_test.columns = columns_names
            dataframe_train.columns = columns_names
            
            try:
                dataframe_train.to_csv(pathToDataset +"/"+ nameDataset_train)
                dataframe_test.to_csv(pathToDataset +"/"+ nameDataset_test)
            except FileNotFoundError:
                dataframe_train.to_csv(pathToDatasetAbs +"/"+ nameDataset_train)
                dataframe_test.to_csv(pathToDatasetAbs +"/"+ nameDataset_test)
                
        print("End creation split, time: {} [s]".format((time.time() -startTime), ))
        
    def loadSplittedDataset(self):
        try:
            train_data = pd.read_csv(pathToDataset + '/' + nameDataset_train).iloc[:,1:]
            test_data = pd.read_csv(pathToDataset + '/' + nameDataset_test).iloc[:,1:]
        except FileNotFoundError:
            try:
                train_data = pd.read_csv(pathToDatasetAbs+ '/' + nameDataset_train).iloc[:,1:]
                test_data = pd.read_csv(pathToDatasetAbs + '/' + nameDataset_test).iloc[:,1:]
            except FileNotFoundError:
                self.splitDataset(True)
                return (self.trainSet,self.testSet)
        self.trainSet = np.array(train_data)
        self.testSet = np.array(test_data)
        
        return(self.trainSet,self.testSet)
        
    
    def printInfo(self):
        # general info of the .csv file
        print(" ******************** info ******************** \n")
        self.data.info()
        
        # label for classes
        print("******************** labels ******************** \n")
        print(self.data.columns)
        #print(type(data.columns))
        
        # correlation between classes
        print(" ***************** correlation **************** \n")
        print(self.data.corr())
        
        # graphic correlation 
        # f,ax = plt.subplots(figsize=(30 ,30))
        # sns.heatmap(self.data.corr(),annot = True, linewidths = 0.6, fmt = ".4f", ax=ax)
        # plt.show()
        # show the first ten rows
        
        print(" ***************** 0-10 rows ***************** \n")
        print(self.data[:10])  #data.head(10)


reader = emotionSensorReader()
reader.loadSplittedDataset()