import numpy as np
import pandas as pd 
import re
import gc
import time
from PIL import Image
import torch as T
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# file system paths and file names
pathToDatasetRel = './dataset/nus-wide'
pathToDatasetAbs = '/home/faber/Documents/EAI1/dataset/nus-wide'

nameDataset = 'nus_wid_data.csv'
nameDatasetTrain = 'nus_wid_train.csv'
nameDatasetTest = 'nus_wid_test.csv'

conceptList = 'Concepts81.txt'
imagesFolder = 'images'

testing = False



def myCollate(batch):
   imgs = [item[0] for item in batch]
   target = [item[2] for item in batch]
   target = T.Tensor(target)
   imgs = T.tensor(imgs)
   
   return [imgs,target]

def preprocess(img):
    # histogram equalization, gamma correction, ....

    img = transforms.functional.adjust_gamma(img, 0.8)
    img = transforms.functional.adjust_saturation(img ,saturation_factor=1.2)
    img = transforms.functional.adjust_sharpness(img,sharpness_factor=2)
    img = transforms.functional.equalize(img)
 
    return img

class NusWide(Dataset):
    
    def __init__(self, data,transformationImg = None,show = False):
        super().__init__()
        self.data = data
        self.transformationImg = transformationImg
        self.show = show 
        # self.imgPathAbs = pathToDatasetAbs + "/" + imagesFolder
        # self.imgPathRel = pathToDatasetRel + "/" + imagesFolder
        

        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_name = self.data[index][0]
        image_labels = self.data[index][1]
        image_encoding = list(self.data[index][2:])
        
        try:
            img = Image.open(pathToDatasetRel + '/' + image_name).convert('RGB')
        except:
            img = Image.open(pathToDatasetAbs + '/' + image_name).convert('RGB')
        
        # img = np.array(img)
        
        if self.show:
            print(type(img))
            img_tmp = np.array(img)
            plt.imshow(img_tmp)
            plt.show()

        # additional transformation (not strictly needed)
        img = preprocess(img)
        
        if self.show:
            pass
            print(type(img))
            img_tmp = np.array(img)
            plt.imshow(img_tmp)
            plt.show()
            
        
        if not (self.transformationImg == None):
            img = self.transformationImg(img)
            # print("transformation")
                         
        if self.show:
            print(type(img))
            # img_tmp = img.numpy()
            img_tmp = T.movedim(img, 0, 2)
            img_tmp = (img_tmp +1)/2
            plt.imshow(img_tmp)
            plt.show()
        
        self.show = False

        
        image_encoding = T.Tensor(image_encoding)
        
        return [img,image_labels,image_encoding]
            


class NusDatasetReader():

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
        
        # if (self.data):
        #     self.computeDataFrame()
            
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
        
        for val in self.data.values:

            # print(val[2])
            
            type_sample = val[2]
            
        
            elem = [val[0],np.array(val[1])]
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
        
    def stringToList(self, text):
        text = re.sub("[\[\]']", '', text) # left the comma for separation
        text = re.sub(" ", '', text)
        list_text = text.split(',')
        print(list_text)
        return list_text
        
    
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

    # to avoid memory leaks, load separately trainset and testset
    
    def retrieveTrainingSet(self):
        try:
            train_data = pd.read_csv(pathToDatasetRel + '/' + nameDatasetTrain).iloc[:, 1:]
        except FileNotFoundError:
            try:
                train_data = pd.read_csv(pathToDatasetAbs+ '/' + nameDatasetTrain).iloc[:, 1:]
            except FileNotFoundError:
                self.splitDataset(True)
                return (self.trainSet,self.testSet)
            
        # casting all data to string (also the encoding)

        train_data = train_data.to_numpy()

        return train_data
    
    def retrieveTestSet(self):
        try:
            test_data = pd.read_csv(pathToDatasetRel + '/' + nameDatasetTest).iloc[:, 1:]
        except FileNotFoundError:
            try:
                test_data = pd.read_csv(pathToDatasetAbs + '/' + nameDatasetTest).iloc[:, 1:]
            except FileNotFoundError:
                self.splitDataset(True)
                return (self.trainSet,self.testSet)
            
        # casting all data to string (also the encoding)
        test_data = test_data.to_numpy()

        return test_data
        
    def freeSpace(self):
        gc.collect()
        
 
if(testing):
    # ______________________test splitting______________________________
    
    data = NusDatasetReader()
    
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


    

