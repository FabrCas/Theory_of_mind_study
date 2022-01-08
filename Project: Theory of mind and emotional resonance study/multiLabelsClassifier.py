import numpy as np 
import sys 
import time 
import os
import re
import math
import gc 
import torch as T
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from randaugment import RandAugment
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy 

from nusWideDatasetAnalyzer import NusDatasetReader, NusWide
from models import ResNet101
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, accuracy_score




device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

# to finish 

class MLC(T.nn.Module):
    
    
    def __init__(self,args):
        super(MLC,self).__init__()
        
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
        self.sigmoid = T.nn.Sigmoid()
        
        

    def printSummaryNetwork(self, inputShape = (32,3,224,224)):
        summary(self.model, inputShape)
    
    def getDataSetReader(self):
        return self.nus_wide_reader
    
    """"
    Parameters
    ----------
    x: input logits
    y: targets (multi-label binarized vector)
    """
            
    def _computeASL(self,logits, targets, min_clip =1e-6 ):
        loss = 0

    
        # From digits to probabilities
        # x_sigmoid = T.sigmoid(logits)
        
        x_sigmoid = self.sigmoid(logits)
        # clip 0 values to be greater otherwise loss nan
        
        x_sigmoid = T.clamp(x_sigmoid, min= min_clip, max=(1-min_clip))
        
        # T.autograd.set_detect_anomaly(True)
        # with T.no_grad():
        #     x_sigmoid[x_sigmoid==0] = min_clip
        
        
        # Define positive and negative samples 
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        
        # Cross-Entrpy error
        los_pos = targets * T.log(xs_pos.clamp(min= self.eps, max = (1-self.eps) ))
        los_neg = (1 - targets) * T.log(xs_neg.clamp(min= self.eps, max = (1-self.eps) ))
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
        
        # print(loss.shape)
        # loss_ = deepcopy(loss)
        # loss_ = loss_.detach().cpu().numpy()
        
        # print(T.isnan(loss).any())
        if T.isnan(loss).any():
            print("loss nan when min values is {}".format(T.min(x_sigmoid)))
            print("loss nan when max values is {}".format(T.max(x_sigmoid)))
            print(los_pos)
            print(los_neg)
            print(loss)
        return -loss.sum()
    
    def _saveModel(self, epoch, folder = 'models/MLC/'):
        path_save = os.path.join(folder)
        name = 'resNet-'+ str(epoch) +'.ckpt'
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        T.save(self.model.state_dict(), path_save + name)
        
        
    def _compute_mAP(self, output, targets, average = "samples"):
        mAP = 0
        n_labels = targets.shape[1]
        print(n_labels)
        ap = np.zeros(n_labels)
        for class_index in range(n_labels):
            # extract values for label
            output_label = output[:,class_index] ; target_label = targets[:,class_index]
            # average precision
            ap[class_index] = average_precision_score(target_label, output_label, average=average)

        where_nans = np.isnan(ap)
        for val in where_nans:
            if val== True:
                print("found nan value, substitution")
                ap[where_nans] = 1
                break

        mAP = np.mean(ap)
        return mAP
        
    def _computeMetrics(self, output,targets, labels, average = "samples"):

        print(output.shape)
        print(targets.shape)
        print(labels.shape)
        # print(targets[0])
        # print(labels)
        
        metrics_results = {
                            # "accuracy": accuracy_score(targets, output),
                            "precision":precision_score(targets, output, average = average, zero_division=1),  \
                            "recall": recall_score(targets, output, average = average, zero_division=1), \
                            "f1-score": f1_score(targets, output, average= average, zero_division=1), \
                            "average precision": average_precision_score(targets, output, average= average), \
                            "mean average precision": self._compute_mAP(output,targets, average)
            }
        
        return metrics_results
            
    def loadModel(self, epoch, test_number = 1):
        path_load = os.path.join('models/MLC_' + str(test_number)+ '/', 'resNet-{}.ckpt'.format(epoch))
        ckpt = T.load(path_load)
        self.model.load_state_dict(ckpt)
        
        
    
        
    def train_MLC(self, save_model = True):

        training_data = self.nus_wide_reader.retrieveTrainingSet()
        training_history = []
        
        print("- started training of the model...")
        
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
        
        n_samples = len(training_dataset)
        
        print("number of samples per epoch (no batch): {}".format(n_samples))

        
        loaderTrain = DataLoader(training_dataset, batch_size= self.batchSize,
                                 shuffle= True, num_workers= self.workers,
                                 pin_memory= True)
        
        n_steps = len(loaderTrain)
        print("number of steps per epoch: {}".format(n_steps))
        
        optimizer = Adam(self.model.parameters(), lr = self.lr, weight_decay= 0)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=n_steps, epochs=self.n_epochs,
                                       pct_start=0.3)
        
        scaler = GradScaler()
        
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
        
        self.model.train()
        
        for n_epoch in range(self.n_epochs) : #self.n_epochs
            loss_cumulative = 0
            for index,(images,labels,encoding_labels) in enumerate(tqdm(loaderTrain)):
                # print(index)
            
                T.cuda.empty_cache()
                # img = images[0]
                # print(img)
                # img = T.movedim(img, 0, 2)
                # img = (img +1)/2
                # print(img)
                
                # plt.imshow(img)
                # plt.show()
                
                optimizer.zero_grad()
                
                images = images.to(device)
                encoding_labels = encoding_labels.to(device)
                # print(images.shape)
                # print(images)
                # print(encoding_labels.shape)
                # print(encoding_labels)
    
    
                with autocast():
                    output = self.model.forward(images) 
                    loss = self._computeASL(output,encoding_labels)
                
                
                # loss = loss.items()

    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # loss.backward()
                # optimizer.step()
                
                scheduler.step()
                
                loss_cumulative += loss.cpu().detach().item()
                
                
                # store information
                if index % 100 == 0:
                    training_history.append([n_epoch, index, loss.item()]) #todo enrichment
                    
                if index % 500 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                          .format(n_epoch, self.n_epochs, str(index).zfill(3), str(n_steps-1).zfill(3),
                                  scheduler.get_last_lr()[0], \
                                  loss))
                

            # avg_lossEpoch = (loss_cumulative/math.ceil((n_steps/self.batchSize)))
            avg_lossEpoch = (loss_cumulative)/n_steps
            print("\naverage loss in batch for this epoch: -> {:.2f}".format(avg_lossEpoch))
            loss_cumulative = 0
            
            if (n_epoch+1)%10 == 0 and save_model:
                self._saveModel(n_epoch+1)
                
                
    def continue_training(self, srcModel , epochs, save_model = True):
        if not os.path.exists(srcModel):
            raise NameError("path doesn't exist!")

        test_n, epoch_loaded = re.findall(r'\d+',srcModel)
        test_n = int(test_n); epoch_loaded = int(epoch_loaded)
        final_epoch = epoch_loaded + epochs
        new_path = "models/MLC_" +str(test_n+1) + "/"
        if not os.path.exists(new_path):
            os.mkdir(new_path)
            
        # load the model
        self.loadModel(epoch_loaded, test_n)
            
            
        training_data = self.nus_wide_reader.retrieveTrainingSet()             # limited !
        training_history = []
        
        print("- started the continue of training for the model, from epoch {} to {}".format(epoch_loaded, final_epoch))
        
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
        del training_data
        
        loaderTrain = DataLoader(training_dataset, batch_size= self.batchSize,
                                 shuffle= True, num_workers= self.workers,
                                 pin_memory= True)
        
        n_steps = len(loaderTrain)
        
        optimizer = Adam(self.model.parameters(), lr = self.lr, weight_decay= 0)
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=n_steps, epochs= epochs,
                                       pct_start=0.3)
        
        scaler = GradScaler()
        
        gc.collect()
        
        self.model.train()
        
        for n_epoch in range(epochs) : 
            loss_cumulative = 0
            for index,(images,labels,encoding_labels) in enumerate(tqdm(loaderTrain)):

            
                T.cuda.empty_cache()

                
                optimizer.zero_grad()
                
                images = images.to(device)
                encoding_labels = encoding_labels.to(device)
    
                with autocast():
                    output = self.model.forward(images) 
                    loss = self._computeASL(output,encoding_labels)
                
            
                
                scaler.scale(loss).backward()
                T.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) #0.25
                scaler.step(optimizer)
                scaler.update()


                
                loss_cumulative += loss.cpu().detach().item()

                scheduler.step()
                
                # store information
                if index % 100 == 0:
                    training_history.append([n_epoch, index, loss.item()]) #todo enrichment
                    
                if index % 500 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                          .format(n_epoch, epochs, str(index).zfill(3), str(n_steps-1).zfill(3),
                                  scheduler.get_last_lr()[0], \
                                  loss))
                # if index == 100:
                #     break
                

            # avg_lossEpoch = (loss_cumulative/math.ceil((n_steps/self.batchSize)))
            avg_lossEpoch = (loss_cumulative)/n_steps
            print("\naverage loss in batch for this epoch: -> {:.2f}".format(avg_lossEpoch))
            loss_cumulative = 0
            
            if (n_epoch+1)%1 == 0 and save_model:   # to edit
                self._saveModel(n_epoch+epoch_loaded, new_path)
            
        
    
    def validate_MLC(self, threshold_truth = 0.5):
        validate_data = self.nus_wide_reader.retrieveTestSet()
        
        print("- started validation of the model...")
        
        from_pos_to_label = self.nus_wide_reader.getLabels()
        

        
        validate_dataset = NusWide(validate_data,
                                   transformationImg= transforms.Compose([
                                       transforms.Resize((224,224)),
                                       # RandAugment(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ]),
                                   transformationLab= transforms.Compose([
                                       transforms.ToTensor()
                                       ])
                                   )

        del validate_data
        
        
        loaderVal = DataLoader(validate_dataset, batch_size= self.batchSize,
                                 shuffle= False, num_workers= self.workers,
                                 pin_memory= True)
        

        
        self.model.eval()
        

        predictions = []; targets = []
        
        for index,(images,labels,encoding_labels) in enumerate(tqdm(loaderVal)):
            
            T.cuda.empty_cache()

            
            if False:
                img = images[0]
                img = T.movedim(img, 0, 2)
                # from [-1,1] to [0,1]
                img = (img +1)/2
                
                plt.imshow(img)
                plt.show()
            
            # images from CPU to GPU
            images = images.to(device)
            
            encoding_labels = encoding_labels.numpy()

                
            with T.no_grad():
                with autocast():
                
                    output_prob  = self.sigmoid(self.model(images)).cpu().detach().numpy()
                    print(np.min(output_prob))

                    temp_labels = []
                    temp_targets = []
                    if False:
                        for i,val in enumerate(output_prob[0]):
                            if val > threshold_truth:
                                temp_labels.append(from_pos_to_label[i])
                        print("")
                        print(temp_labels)
                        
                        for i,val in enumerate(encoding_labels[0]):
                            if val == 1:
                                temp_targets.append(from_pos_to_label[i])
                                
                        print(temp_targets)
                    
                    # pass from probability to binary classification
                    output_bin = np.array(output_prob > threshold_truth)
                    
                    # print(output_bin.shape)
                    # print(encoding_labels.shape)
                    
                    
                    predictions.append(output_bin)
                    targets.append(encoding_labels)
                

            # if index == 100:
            #     break
            
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        print("\n\n\n")
        
        
        # flat to 1-D
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        evaluations = self._computeMetrics(predictions,targets, from_pos_to_label)
        print(evaluations)
        
        
    # {'precision': 0.5219604682252553, 'recall': 0.7361859292236855, 'f1-score': 0.5728981285236443, 'average precision': 0.45115118659603126}        
            
            
               
            
            
            
            
        
        
c = MLC(1)
if False:
    # c.train_MLC()
    c.loadModel(epoch= 39, test_number= 2)
    # c.printSummaryNetwork( (3,224,224) )
    c.validate_MLC()
    
c.continue_training("models/MLC_2/resNet-30.ckpt", 20)




    
