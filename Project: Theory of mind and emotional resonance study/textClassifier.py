import numpy as np 
import sys 
import time 
import os
import re
import math
import gc 
import torch as T
from emotionDatasetAnalyzer import emotionSensorReader
import torchtext as TT




device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

class TC():
    def __init__(self, args):
        self.emo_sensor_reader = emotionSensorReader()
        self.trainSet, self.testSet = self.emo_sensor_reader.loadSplittedDataset()
        # print(self.trainSet)
        # print(self.testSet)
        
        
    def _embedding(self):
            # glove = TT.vocab.GloVe(name="6B", dim=100)
            glove = TT.vocab.GloVe(name="840B", dim=300)
            
            # t1 =  glove['cat'].unsqueeze(0)
            # t2 =   glove['dog'].unsqueeze(0)
            
            # print(T.cosine_similarity(t1,t2))
        
    def saveModel(self):
        pass
    
    def loadModel(self):
        pass
    
    def train_TC(self, save_model = True):
        pass
    
    def validate_TC(self):
        pass
    
    

new = TC(0)
new._embedding()