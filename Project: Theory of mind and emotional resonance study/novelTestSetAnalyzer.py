import numpy as np
import pandas as pd 
import re
import gc
import time
import os



pathToDataset = './dataset/novel_testset'
pathToDatasetAbs = "/home/faber/Desktop/ProjectCode/Project: Theory of mind and emotional resonance study/dataset/novel_testset"

class novelTestSet():
    def __init__(self):
        self.dataframes = {}
        self.images = {}
        self.audioRecords = {}
        self.groundTruth = {}
        self._extractElements(print_info=True)
        
        
    def getLabelsGT(self):
        tmp_dataframe = pd.read_csv('./dataset/novel_testset/record_1/gt.csv', sep =',')
        columns = np.array(tmp_dataframe.columns.tolist())[1:]
        return columns
        
    def _extractElements(self, print_info = False):
        
        def order_rule(elem):
            elems = elem.split('_')
            return int(elems[1])
        
        try:
            samples = os.listdir(pathToDataset)
            directory = pathToDataset
        except:
            samples = os.listdir(pathToDatasetAbs)
            directory = pathToDatasetAbs
        
        samples = sorted(samples, key=order_rule)
            
        for sample in samples:
            if print_info: print("\n----------------[{}]-----------------\n".format(sample))
            
            path = os.path.join(directory,sample)
            if print_info: print(path)
            tmp_dataframe = pd.read_csv(path + '/gt.csv' )
            if print_info: print(tmp_dataframe)
            
            self.dataframes[sample] = tmp_dataframe
            images_list = sorted(os.listdir(path+"/images"))
            audio_list = [path +"/"+ str(i+1) + ".wav" for i in range(len(images_list))]
        
            if print_info: print(audio_list)
            if print_info: print(images_list)
            self.images[sample] = images_list
            self.audioRecords[sample] = audio_list
            
            # GT to array
            values = np.zeros((5,7))
            tmp_dataframe = tmp_dataframe.iloc[1:,1:] 
            # print(tmp_dataframe)
            for index,val in enumerate(tmp_dataframe.values):

                val = np.array(val)
                values[index] = val
            
            print(values)
            if print_info: self.groundTruth[sample] = values

            
# -------------------------------[test section]

test = novelTestSet()
# print(test.getLabelsGT())
# test.getElements(print_info = True)
    