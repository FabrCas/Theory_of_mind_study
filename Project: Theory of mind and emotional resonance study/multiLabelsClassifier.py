import numpy as np 
import torch as T
from torchsummary import summary
import sys 
import time 
import os
from nusWideDatasetAnalyzer import NusDataset
from models import ResNet101


device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

# to finish 

class MLCClassifier(T.nn.Module):
    
    
    def __init__(self,args):
        super(MLCClassifier,self).__init__()
        
        # self.lr = args.lr
        # self.epoch = args.epoch
        # self.batchSize = args.batchSize
        # todo
        
        self.gamma_neg=4
        self.gamma_pos=1
        self.clip=0.05
        self.eps=1e-8
        self.disable_torch_grad_focal_loss=True
        
        self.nus_wide_dataset = NusDataset()
        
        # retrieve test and training set
        # nus_wide_dataset.splitDataset(False) 
        # self.trainSet = nus_wide_dataset.trainSet
        # self.testSet = nus_wide_dataset.testSet
        # nus_wide_dataset.freeSpace()
        
        # create the RNN
        self.model = ResNet101()
        
        
    """"
    Parameters
    ----------
    x: input logits
    y: targets (multi-label binarized vector)
    """
            
    def computeASL(self,logits, targets):
    
        # Calculating Probabilities
        x_sigmoid = T.sigmoid(logits)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        # Cross-Entrpy error
        los_pos = targets * T.log(xs_pos.clamp(min= self.eps))
        los_neg = (1 - targets) * T.log(xs_neg.clamp(min= self.eps))
        loss = los_pos + los_neg
        
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                T.set_grad_enabled(False)
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w = T.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                T.set_grad_enabled(True)
            loss *= one_sided_w
        
        return -loss.sum()
    

    def printSummaryNetwork(self):
        summary(self.model)
    
    def getDataSet(self):
        return self.nus_wide_dataset
        
    def train_MLC(self):
        training_data = self.nus_wide_dataset.retrieveTrainingSet()
    
    def test_MLC(self):
        test_data = self.nus_wide_dataset.retrieveTestSet()
    



    
