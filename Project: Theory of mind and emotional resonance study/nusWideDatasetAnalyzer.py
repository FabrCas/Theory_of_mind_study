import numpy as np
import pandas as pd 
import re
import gc
import time

# file system paths and file names
pathToDatasetRel = './dataset/nus-wide'
pathToDatasetAbs = '/home/faber/Documents/EAI1/dataset/nus-wide'

nameDataset = 'nus_wid_data.csv'
nameDatasetTrain = 'nus_wid_train.csv'
nameDatasetTest = 'nus_wid_test.csv'

conceptList = 'Concepts81.txt'
imagesFolder = 'images'

testing = False


class NusDataset():

    def __init__(self):
        self.data = None
        self.labels = np.array([[]]) 
        
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
    
    
    
    def getLabels(self, printIt = False):
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
        if (printIt): print(labels)
        self.labels = labels 
        return labels
            
    
    def extractLabels(self, printIt = False):
        
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
        if (printIt): print(labels)
        self.labels = labels
        return labels
        
        # print(typer)
        
    def createEncode(self,labels):
        if (self.labels.size == 0): self.getLabels()
        
        # parsing the string of labels to a list
        
        labels = re.sub("[\[\]']", '', labels) # left the comma for separation
        labels = re.sub(" ", '', labels)
        labels = labels.split(',')
        
        # print(type(labels))
        # print(labels)
        
        
        # encoding = [[0]] *  len(self.labels)
        encoding = [0] *  len(self.labels)
        test_list = []
        for index,label_1 in enumerate(self.labels):
            
            for label_2 in labels:
                if(label_1 == label_2):

                    test_list.append(label_1)
                    # encoding[index] = [1]
                    encoding[index] = 1
        if not(test_list == labels):
            print(test_list)
            print(labels)
            raise ValueError('Error during encoding')

        # print(encoding)
        return encoding
        
        
            
    
    def splitDataset(self, writeFiles = False):

          
        train_list = []
        test_list = []
        
        startTime = time.time()
        
        # counter_train = 0
        # conuter_val = 0
        # counter_test = 0
        # train = []
        # test = []
        # validation = []`
        
        for val in self.data.values:     # limit to 15 (test)

            # print(val[2])
            
            type_sample = val[2]
            
        
            elem = [val[0],val[1]]
            # print(elem)
            temp_encoding = self.createEncode(val[1])
            # print(temp_encoding)
            elem = elem + temp_encoding
            # elem = [*elem,*temp_encoding]

            if (type_sample == "train"):                
                train_list.append(elem)

                
            elif(type_sample == "val"):
                test_list.append(elem)
        
        
        # check this: converting from python list to numpy all the types of data become String
        # try to convert it here or handle this directly in the main
        
        self.trainSet = np.array(train_list)
        self.testSet = np.array(test_list)
        # print(self.trainSet)
        
        
        # do and add  multiple encoding for labels
        
        
        if(writeFiles):
            columns_names =  []
            columns_names = self.labels
            
            columns_names = np.insert(columns_names,0,'labels')
            columns_names = np.insert(columns_names,0,"image_name")
            
            # print(columns_names)
            dataframe_train = pd.DataFrame(self.trainSet)
            dataframe_test =  pd.DataFrame(self.testSet)
            dataframe_test.columns = columns_names
            dataframe_train.columns = columns_names
            
            try:
                dataframe_train.to_csv(pathToDatasetRel +"/"+ nameDatasetTrain)
                dataframe_test.to_csv(pathToDatasetRel +"/"+ nameDatasetTest)
            except FileNotFoundError:
                dataframe_train.to_csv(pathToDatasetAbs +"/"+ nameDatasetTrain)
                dataframe_test.to_csv(pathToDatasetAbs +"/"+ nameDatasetTest)
                
        print("End creation split, time: {} [s]".format((time.time() -startTime), ))
        
    
    
    def loadSplittedDataset(self):
        try:
            train_data = pd.read_csv(pathToDatasetRel + '/' + nameDatasetTrain).iloc[:, 1:]
            test_data = pd.read_csv(pathToDatasetRel + '/' + nameDatasetTest).iloc[:, 1:]
        except FileNotFoundError:
            try:
                train_data = pd.read_csv(pathToDatasetAbs+ '/' + nameDatasetTrain).iloc[:, 1:]
                test_data = pd.read_csv(pathToDatasetAbs + '/' + nameDatasetTest).iloc[:, 1:]
            except FileNotFoundError:
                self.splitDataset(True)
                return (self.trainSet,self.testSet)
            
        # casting all data to string (also the encoding)
        train_data = train_data.to_numpy(str)
        test_data = test_data.to_numpy(str)
        self.trainSet = train_data
        self.testSet = test_data
        return(self.trainSet,self.testSet)
        
    
    def freeSpace(self):
        gc.collect()
        
        
        
if(testing):
    # ______________________test splitting______________________________
    
    data = NusDataset()
    
    data.splitDataset(True)
    # print(data.testSet)
    data.getLabels(False)
    
    
    
    # print(type(data.data.values))
    print(len(data.data.values))
    print(len(data.trainSet))
    print(len(data.testSet))
    
    print("---------------------------------------------------------\n")
    print(data.trainSet)
    print(type(data.trainSet))
    print("---------------------------------------------------------\n")
    print(data.testSet)
    
    # tr,te = data.loadSplittedDataset()
    # print("---------------------------------------------------------\n")
    # print(tr)
    # print(type(tr))
    # print("---------------------------------------------------------\n")
    # print(te)
    
    
    print("---------------------------------------------------------\n")
    
    
    # ______________________test encoding_________________________________
    
    
    print(data.trainSet[0])
    
    data.createEncode(data.trainSet[0][1])


    

