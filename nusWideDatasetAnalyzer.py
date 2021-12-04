import numpy as np
import os
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import re
import gc
import time


pathToDatasetRel = './dataset/nus-wide'
pathToDatasetAbs = '/home/faber/Documents/EAI1/dataset/nus-wide'

nameDataset = 'nus_wid_data.csv'
nameDatasetTrain = 'nus_wid_train.csv'
nameDatasetTest = 'nus_wid_test.csv'

conceptList = 'Concepts81.txt'
imagesFolder = 'images'




class NusDataset():

    def __init__(self):
        self.data = None
        self.labels = None
        
        # division of the dataset: approximately 70% train and 30% test
        
        self.trainSet = np.array([[]])   
        self.testSet = np.array([[]])
        
        self.computeDataFrame()
        
     
    # read and store data from csv file
    
    def computeDataFrame(self):
        try:
            data = pd.read_csv(pathToDatasetRel + '/' + nameDataset)
        except FileNotFoundError:
            data = pd.read_csv(pathToDatasetAbs + '/' + nameDataset)
            
        self.data = data 
    
    
    
    def getLabels(self):
        labels = np.array([])
        try:
            with open(pathToDatasetRel + '/' + conceptList, "r") as file:
                for line in file.readlines():
                    line = re.sub('[\n]$', '', line)
                    labels = np.append(labels,[line], axis=0)
                file.close()
                
        except FileNotFoundError: 
            with open(pathToDatasetAbs + '/' + conceptList, "r") as file:
                for line in file.readlines():
                    line = re.sub('[\n]$', '', line)
                    labels = np.append(labels,[line], axis=0)
                file.close()
        print(labels)
        self.labels = labels 
            
    
    def extractLabels(self):
        
        if (self.data):
            self.computeDataFrame()
            
        # final list of labels extracted 
        labels = []

        # typer = []
        
        #  transform dataframe into numpy array and iterate 
        for val in self.data.values:
            
            # string with the labels
            temp_labels = val[1]
            # temp_type = str(val[2])
            
            # clear the labels and split them
            temp_labels = re.sub("[\[\]']", '', temp_labels) # left the comma for separation
            temp_labels = re.sub(" ", '', temp_labels)
            
            temp_labels = temp_labels.split(',')
            
            # add label and check if not present
            [labels.append(x) for x in temp_labels if not(x in labels)]
            
            # if not(temp_type in typer): 
            #     typer.append(temp_type) 
            
        labels.sort()
        print(labels)
        self.labels = labels
        
        # print(typer)
        
            
    
    def splitDataset(self):

        writeFiles = False   
        train_list = []
        test_list = []
        
        startTime = time.time()
        
        # counter_train = 0
        # conuter_val = 0
        # counter_test = 0
        # train = []
        # test = []
        # validation = []
        
        for val in self.data.values:     # limit to 15 (test)

            # print(val[2])
            
            type_sample = val[2]
            elem = [val[0],val[1]]
            
            
            if (type_sample == "train"):
                train_list.append(elem)
                
            elif(type_sample == "val"):
                test_list.append(elem)
            
        self.trainSet = np.array(train_list)
        self.testSet = np.array(test_list)
        
        
        # do and add  multiple encoding for labels
        
        
        if(writeFiles):
            try:
                pd.DataFrame(self.trainSet).to_csv(pathToDatasetRel +"/"+ nameDatasetTrain)
                pd.DataFrame(self.testSet).to_csv(pathToDatasetRel +"/"+ nameDatasetTest)
            except FileNotFoundError:
                pd.DataFrame(self.trainSet).to_csv(pathToDatasetAbs +"/"+ nameDatasetTrain)
                pd.DataFrame(self.testSet).to_csv(pathToDatasetAbs +"/"+ nameDatasetTest)
                
        print("End creation split, time: {} [s]".format((time.time() -startTime), ))
        
            
    def freeSpace(self):
        gc.collect()
        
        
        
        
# _____________________________test______________________________

data = NusDataset()

data.splitDataset()
# print(data.testSet)


# print(type(data.data.values))
print(len(data.data.values))


print(len(data.trainSet))
# print(data.trainSet[0])
# print(data.trainSet[1])

print(len(data.testSet))
# print(data.testSet[0])
# print(data.testSet[1])


    

