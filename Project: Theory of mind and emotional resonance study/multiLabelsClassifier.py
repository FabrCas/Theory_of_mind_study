import numpy as np 
import sys 
import time 
import os
import gc 
import torch as T
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from randaugment import RandAugment
import matplotlib.pyplot as plt

from nusWideDatasetAnalyzer import NusDatasetReader, NusWide
from models import ResNet101




device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

# to finish 

class MLCClassifier(T.nn.Module):
    
    
    def __init__(self,args):
        super(MLCClassifier,self).__init__()
        
        # self.lr = args.lr
        # self.epoch = args.epoch
        # self.batchSize = args.batchSize
        
        self.batchSize = 32 #greater batch size lead to an over allocation error
        self.workers = 8
        self.lr = 1e-4
        self.n_epochs = 50 
        
        # todo
        
        self.gamma_neg=4
        self.gamma_pos=1
        self.clip=0.05
        self.eps=1e-8
        self.disable_torch_grad_focal_loss=True
        
        self.nus_wide_reader = NusDatasetReader()
        
        # retrieve test and training set
        # nus_wide_dataset.splitDataset(False) 
        # self.trainSet = nus_wide_dataset.trainSet
        # self.testSet = nus_wide_dataset.testSet
        # nus_wide_dataset.freeSpace()
        
        # create the RNN
        self.model = ResNet101()
        # self.model.parameters().half()
        self.model.to(device)
        
        
        
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
    
    def getDataSetReader(self):
        return self.nus_wide_reader
    
    
        
        
    def train_MLC(self):

        training_data = self.nus_wide_reader.retrieveTrainingSet()[0:1000]                           # limited !
        
        
        
        training_dataset = NusWide(training_data,
                                   transformationImg= transforms.Compose([
                                       transforms.Resize((224,224)),
                                       RandAugment(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ]),
                                   transformationLab= transforms.Compose([
                                       transforms.ToTensor()
                                       ])
                                   )
        """
        image = (image - mean) / std   -> range [-1,1]
        image = ((image * std) + mean) -> range [0,1]
        """
        del training_data
        
        n_steps = len(training_dataset)
        
        print("number of steps: {}".format(n_steps))

        
        loaderTrain = DataLoader(training_dataset, batch_size= self.batchSize,
                                 shuffle= True, num_workers= self.workers,
                                 pin_memory= True)
        
        optimizer = Adam(self.model.parameters(), lr = self.lr, weight_decay= 0)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=n_steps, epochs=self.n_epochs,
                                       pct_start=0.3)
        
        if False:
            iterdata = next(iter(loaderTrain))
            
            images = iterdata[0]
            labels = iterdata[1]
            targets = iterdata[2]
            print(images.shape)
            print(targets.shape)
            
            print(type(images[0]))
            print(type(images[2]))
            
            print(images[0])
            print(labels[0])
            print(targets[0])
            img = images[0]
            
    
            img = T.movedim(img, 0, 2)
            print(img.shape)
            img = (img +1)/2 # move the nomalized interval [0,1]
            plt.imshow(img)
            plt.show()
        
        gc.collect()
        
        # for n_epoch in range(self.n_epochs) :
        for index,(images,labels,encoding_labels) in enumerate(loaderTrain):
            print(index)
        
            T.cuda.empty_cache()
            # img = images[0]
            # print(img)
            # img = T.movedim(img, 0, 2)
            # img = (img +1)/2
            # print(img)
            
            # plt.imshow(img)
            # plt.show()
            
            images = images.to(device)


            output = self.model.forward(images)
            
            # optimizer.zero_grad()
            # loss = 10
            # loss.backward()
            # optimizer.step()
            # scheduler.step()
            
            
            
 
        
        
        
    def test_MLC(self):
        test_data = self.nus_wide_reader.retrieveTestSet()
    

c = MLCClassifier(1)
c.train_MLC()

    
