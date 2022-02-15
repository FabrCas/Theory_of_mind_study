import os
import numpy as np 
import gc 
import torch as T
from PIL import Image
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.cuda.amp import GradScaler, autocast
from randaugment import RandAugment
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score,     \
    average_precision_score, multilabel_confusion_matrix, hamming_loss,  \
    jaccard_score, label_ranking_loss
    
from nusWideDatasetAnalyzer import NusDatasetReader, NusWide, preprocess
from models import ResNet101

device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

# to finish 

class MLC(T.nn.Module):
    
    
    def __init__(self,args):
        super(MLC,self).__init__()
        
        # self.lr = args.lr
        # self.epoch = args.epoch
        # self.batchSize = args.batchSize
        
        self.batchSize = 32 # greater batch size lead to an over allocation error given the huge n of params for the net
        self.workers = 8
        self.lr = 1e-4
        self.n_epochs = 50 
        
        # todo
        
        self.gamma_neg=4
        self.gamma_pos=1
        self.clip=0.05
        self.eps=1e-8
        self.disable_torch_grad_focal_loss=True
        
        # create dataset reader
        self.nus_wide_reader = NusDatasetReader()

        
        # create the RNN
        self.model = ResNet101()
        
        self.model.to(device)
        self.sigmoid = T.nn.Sigmoid()
        self.threshold_truth = 0.5
        
        # trasformation input
        self.toModel_transf = transforms.Compose([
            transforms.Resize((224,224)),
            # RandAugment(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        

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
        
    def loadModel(self, epoch = 70, test_number = "01"):
        path_load = os.path.join('models/MLC_' + str(test_number)+ '/', 'resNet-{}.ckpt'.format(epoch))
        ckpt = T.load(path_load)
        self.model.load_state_dict(ckpt)
        
    def _plot_cm(self,cm, label):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.matshow(cm, cmap=plt.cm.Greens, alpha=0.5)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x= j, y = i, s=cm[i, j], va='center', ha='center', size='xx-large')
         
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Targets', fontsize=18)
        plt.title('Confusion Matrix '+ label, fontsize=18)
        plt.savefig("./results/cm_"+label+".png")
        plt.show()
    
    def _compute_cms(self,targets,output,labels, plot_and_save = True):
                
        cms = multilabel_confusion_matrix(targets,output,samplewise = False)
        cms_normalized = {}
        for index,cm in enumerate(cms):

            cm_norm= cm/np.sum(cm, axis=1)[:, np.newaxis]
            cms_normalized[labels[index]] = np.round(cm_norm,4)
            if plot_and_save:
                self._plot_cm(cm_norm,labels[index])          
        
        return cms_normalized
                
        
    def _computeMetrics(self,targets, output, labels, average = "samples", save_results = True):  #labels not used

        # print(output.shape)
        # print(targets.shape)
        # print(labels.shape)

        metrics_results = {
                            "precision":precision_score(targets, output, average = average, zero_division=1),    \
                            "recall": recall_score(targets, output, average = average, zero_division=1),         \
                            "f1-score": f1_score(targets, output, average= average, zero_division=1),            \
                            "average_precision": average_precision_score(targets, output, average= average),     \
                            "ranking_loss": label_ranking_loss(targets, output),                                 \
                            "hamming_loss": hamming_loss(targets,output),                                        \
                            "jaccard_score": jaccard_score(targets, output, average= average, zero_division=1),  \
                            "confusion matrices": self._compute_cms(targets, output, labels, save_results)
     
            }
        
        if save_results: np.save("./logs/test_results.npy",metrics_results)
        
        for k,v in metrics_results.items():
            if k != "confusion matrices":
                print("\nmetric: {}, result: {}".format(k,v))
            else:
                for kcm,vcm in v.items():
                    print("\nconfusion matrix for: {}".format(kcm))
                    print(vcm)
                    
        return metrics_results
            

        
    def _logs(self, loss_history, performance_history):
        
        # path for log 
        path_trainLogs = "./logs"
        
        path_loss = path_trainLogs + "/loss_history.npy"
        path_perf = path_trainLogs + "/performance_history.npy"
    
        np.save(path_loss, loss_history)
        np.save(path_perf, performance_history)
        
        
    def _plots(self):
        path_trainLogs = "./logs"
        path_results = "./results"
        
        path_loss = path_trainLogs + "/loss_history.npy"
        path_perf = path_trainLogs + "/performance_history.npy"
        
        if not os.path.exists(path_results):
            os.makedirs(path_results)
        
        # load loss data 
        
        loss_history = np.load(path_loss)
        
        # loss plot 
        
        x = [x[0]+1 for x in  loss_history]
        y = [x[1] for x in  loss_history]
        plt.figure()
        plt.xlabel("Epochs")
        plt.ylabel("Average Loss over epochs")
        plt.title("Training Loss")
        plt.plot(x,y, color = "red")
        plt.savefig(path_results + "/loss_history.png")
        plt.show()
        plt.close()
        
        # load performance data 
        
        performance_history = np.load(path_perf, allow_pickle= True)
        # print(performance_history.shape)
        # print(performance_history)
        
        # extract performance data
        
        y_precision = [x['precision'] for x in  performance_history]
        y_recall = [x['recall'] for x in  performance_history]
        y_f1 = [x['f1-score'] for x in  performance_history]
        y_ap = [x['average precision'] for x in  performance_history]
        y_map = [x['mean average precision'] for x in  performance_history]
        
        x = [n+1 for n in range(len(performance_history))]
        
        # plots
        # --- precision
        plt.figure()
        plt.xlabel("Epochs")
        plt.ylabel("Precision")
        plt.title("Training Precision")
        plt.plot(x, y_precision)
        plt.savefig(path_results + "/precision_history.png")
        plt.show()
        plt.close()
        
        # --- recall
        plt.figure()
        plt.xlabel("Epochs")
        plt.ylabel("Recall")
        plt.title("Training Recall")
        plt.plot(x, y_recall)
        plt.savefig(path_results + "/recallhistory.png")
        plt.show()
        plt.close()
        
        # --- f1-score
        plt.figure()
        plt.xlabel("Epochs")
        plt.ylabel("F1-score")
        plt.title("Training F1-score")
        plt.plot(x, y_f1)
        plt.savefig(path_results + "/f1_history.png")
        plt.show()
        plt.close()
        
        # --- average precision
        
        plt.figure()
        plt.xlabel("Epochs")
        plt.ylabel("Average Precision")
        plt.title("Training Average Precision")
        plt.plot(x, y_ap)
        plt.savefig(path_results + "/ap_history.png")
        plt.show()
        plt.close()
        
        # --- mean average precision
        
        plt.figure()
        plt.xlabel("Epochs")
        plt.ylabel("MAP")
        plt.title("Training Mean Average Precision")
        plt.plot(x, y_map)
        plt.savefig(path_results + "/map_history.png")
        plt.show()
        plt.close()
        

        
    def train_MLC(self, n_test, save_model = True):

        from_pos_to_label = self.nus_wide_reader.getLabels()
        training_data = self.nus_wide_reader.retrieveTrainingSet()
        training_history = [] ; training_performance = []
        srcModel = "models/MLC_0" + n_test + "/"
        
        print("- started training of the multi labels classifier...")
        
        training_dataset = NusWide(training_data,
                                   transformationImg= transforms.Compose([
                                       transforms.Resize((224,224)),
                                       RandAugment(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ]), show= False 
                                   # transformationLab= transforms.Compose([
                                   #     transforms.ToTensor()
                                   #     ])
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
    
        # to remove
        if False:
            iterdata = next(iter(loaderTrain))
            print(len(iterdata))
            print(len(iterdata[0]))
            
            images = iterdata[0]
            labels = iterdata[1]
            targets = iterdata[2]

            img = images[0]
            
            img = T.movedim(img, 0, 2)
            # print(img.shape)
            img = (img +1)/2 # move the nomalized interval [0,1]
            plt.imshow(img)
            plt.show()
        
        gc.collect()
        
        self.model.train()

        
        for n_epoch in range(self.n_epochs) : #self.n_epochs
            loss_cumulative = 0
            for index,(images,labels,encoding_labels) in enumerate(tqdm(loaderTrain)):
            
                T.cuda.empty_cache()
                
                optimizer.zero_grad()
                
                images = images.to(device)
                encoding_labels = encoding_labels.to(device)
    
    
                with autocast():
                    output = self.model.forward(images) 
                    loss = self._computeASL(output,encoding_labels)
                
                
                # loss = loss.items()

                scaler.scale(loss).backward()
                T.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                # loss.backward()
                # optimizer.step()
                
                scheduler.step()
                
                loss_cumulative += loss.cpu().detach().item()
                
                
                # store information
                # if index % 100 == 0:
                #     training_history.append([n_epoch, index, loss.item()]) #todo enrichment
                    
                if index % 500 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                          .format(n_epoch, self.n_epochs, str(index).zfill(3), str(n_steps-1).zfill(3),
                                  scheduler.get_last_lr()[0], \
                                  loss))
                
                # if index > 50:
                #     break
                
            avg_lossEpoch = (loss_cumulative)/n_steps
            
            training_history.append([n_epoch,avg_lossEpoch])
            
            # sample from last step the output and measure the performance
            
            output_bin = np.array(output.cpu().detach().numpy() > self.threshold_truth)
            targets = encoding_labels.cpu().detach().numpy()
            
            training_performance.append(self._computeMetrics(output_bin, targets, from_pos_to_label))
            

            print("\naverage loss in batch for this epoch: -> {:.2f}".format(avg_lossEpoch))
            loss_cumulative = 0
            
            if (n_epoch+1)%5 == 0 and save_model:
                self._saveModel(n_epoch+1, srcModel)
            
            # logs and results save
            self._logs(training_history, training_performance)
            self._plots()
                
                
    def continue_training(self, n_test = "01", epochs = 50, epoch_loaded = 20,save_model = True):

            
        from_pos_to_label = self.nus_wide_reader.getLabels()
        path_trainLogs = "./logs"
        path_loss = path_trainLogs + "/loss_history.npy"
        path_perf = path_trainLogs + "/performance_history.npy"
        srcModel = "models/MLC_0" + str(n_test) + "/"
        
        if not os.path.exists(srcModel):
            raise NameError("path doesn't exist!")
            
        final_epoch = epoch_loaded + epochs

        training_history = np.load(path_loss)[:epoch_loaded].tolist()
        
        print(training_history)
        
        training_performance = np.load(path_perf, allow_pickle= True)[:epoch_loaded].tolist()
        
        # print(training_performance)
        # print(len(training_performance))
        # print(len(training_history))
        
            
        # test_n, epoch_loaded = re.findall(r'\d+',srcModel)
        # test_n = int(test_n); epoch_loaded = int(epoch_loaded)

        
        # new_path = "models/MLC_" +str(test_n+1) + "/"
        

        
        # if not os.path.exists(new_path):
        #     os.mkdir(new_path)
            
        # load the model
        self.loadModel(epoch_loaded, n_test)
            
            
        training_data = self.nus_wide_reader.retrieveTrainingSet()
        
        print("- started the continue of training for the model, from epoch {} to {}".format(epoch_loaded+1, final_epoch))
        
        training_dataset = NusWide(training_data,
                                   transformationImg= transforms.Compose([
                                       transforms.Resize((224,224)),
                                       RandAugment(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ])
                                   # transformationLab= transforms.Compose([
                                   #     transforms.ToTensor()
                                   #     ])
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
                
                    
                if index % 500 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                          .format(n_epoch+ epoch_loaded +1, epochs + epoch_loaded, str(index).zfill(3), str(n_steps-1).zfill(3),
                                  scheduler.get_last_lr()[0], \
                                  loss))
                # if index == 100:
                #     break
                

            # avg_lossEpoch = (loss_cumulative/math.ceil((n_steps/self.batchSize)))
            avg_lossEpoch = (loss_cumulative)/n_steps

            training_history.append([n_epoch + epoch_loaded,avg_lossEpoch])
            
            # sample from last step the output and measure the performance
            
            output_bin = np.array(output.cpu().detach().numpy() > self.threshold_truth)
            targets = encoding_labels.cpu().detach().numpy()
            
            training_performance.append(self._computeMetrics(output_bin, targets, from_pos_to_label))
            
            print("\naverage loss in batch for this epoch: -> {:.2f}".format(avg_lossEpoch))
            
            loss_cumulative = 0
            
            if (n_epoch+ epoch_loaded+1)%5 == 0 and save_model:   # to edit
                self._saveModel(n_epoch+1+epoch_loaded, srcModel)
            
            self._logs(training_history, training_performance)
            self._plots()
            
        
    def test_MLC(self, load_results = True):
        validate_data = self.nus_wide_reader.retrieveTestSet()
        
        print("- started testing of the model...")
        
        from_pos_to_label = self.nus_wide_reader.getLabels()
        path_testLogs = "./logs"
        path_predictionsTest = path_testLogs + "/test_predictions.npy"
        path_targetsTest = path_testLogs + "/test_targets.npy"

        if not(load_results):
            test_dataset = NusWide(validate_data,
                                       transformationImg= self.toModel_transf
                                       # transformationLab= transforms.Compose([
                                       #     transforms.ToTensor()
                                       #     ])
                                       )
    
            del validate_data
            
            
            loaderVal = DataLoader(test_dataset, batch_size= self.batchSize,
                                     shuffle= False, num_workers= 1, # self.workes
                                     pin_memory= True)
            
            self.model.eval()
            
            test_show_results = False 
    
            # predictions = []; targets = []
            predictions = np.empty((0, 81)); targets = np.empty((0, 81))
            
            for index,(images,labels,encoding_labels) in enumerate(tqdm(loaderVal)):
                
                T.cuda.empty_cache()
    
                
                if test_show_results:
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
    
        
                        if test_show_results:
                            temp_labels = []
                            temp_targets = []
                            for i,val in enumerate(output_prob[0]):
                                if val > self.threshold_truth:
                                    temp_labels.append(from_pos_to_label[i])
                            print("")
                            print(temp_labels)
                            
                            for i,val in enumerate(encoding_labels[0]):
                                if val == 1:
                                    temp_targets.append(from_pos_to_label[i])
                                    
                            print(temp_targets)
                            
                            # if index > 1:
                            #     test_show_results = False
                            #     break
                        
                        # pass from probability to binary classification
                        output_bin = np.array(output_prob > self.threshold_truth)
                        
                        encoding_labels = np.array(encoding_labels == 1)
                        
                        # print(output_bin.shape)
                        # print(encoding_labels.shape)
                        
                        # predictions.append(output_bin)
                        # targets.append(encoding_labels)
                        
                        predictions = np.append(predictions, output_bin, axis  =0)
                        targets = np.append(targets,encoding_labels, axis  =0)
                    
            
            np.save(path_predictionsTest, predictions)
            np.save(path_targetsTest, targets)     
            
        else:
            # load results 
            predictions = np.load(path_predictionsTest)
            targets = np.load(path_targetsTest)
            
            
        if False:
            predictions = np.array(predictions)
            targets = np.array(targets)
            
            print("\n\n\n")
            
            # flat to 1-D
            predictions = np.concatenate(predictions)
            targets = np.concatenate(targets)
            
            evaluations = self._computeMetrics(predictions,targets, from_pos_to_label)
            print(evaluations)
        
        # predictions = np.array(predictions)
        # targets = np.array(targets)
        
        # print(predictions.shape)
        # print(targets.shape)
    
        self._computeMetrics(targets,predictions,from_pos_to_label,average="samples", save_results= False)

    def forward(self,x, show = True):
        
        from_pos_to_label = self.nus_wide_reader.getLabels()
        
        # list for the outcome
        results = []
        
        # pre-processing of the image/s
        x = preprocess(x)
        
        # load the model
        self.loadModel()
        
        if show:
            x_tmp = np.array(x)
            plt.imshow(x_tmp)
            plt.show()
            
        # resize,normalize and trasform to Tensor
        x = self.toModel_transf(x)
        if show:
            img_tmp = T.movedim(x, 0, 2)
            img_tmp = (img_tmp +1)/2
            plt.imshow(img_tmp)
            plt.show()
        
        # add fist dimension and send to GPU
        if len(x.shape) != 4: x = T.unsqueeze(x,axis = 0)
        
        # send to GPU
        x = x.to(device)
        
        # x = T.zeros([1,3,224,224], dtype=T.float32).to(device)
        
        # get probabilities
        y  = self.sigmoid(self.model(x)).cpu().detach().numpy()[0]
        
        # get classification outcome for label
        # the threshold can be reduced to get more labels, i.e: 0.08
        y_bin = np.array(y > self.threshold_truth) 
        
        # manage case for samples with no classification found
        if np.count_nonzero(y_bin == True) == 0:
            print("The image has no recognizable concepts from NusWide dataset")
            max_val = np.max(y)
            index_max = np.where(y == max_val)[0][0]
            print("Anyway, the most probable classification is: {}, with a probability of {}".format(from_pos_to_label[index_max],max_val))
            results.append(from_pos_to_label[index_max])
        else:
            print("Results for the image:")
            for i,val in enumerate(y_bin):
                if val == True:
                    results.append(from_pos_to_label[i])
            print(results)
        
        return results
    
            
    
        
        
        
        
    # epoch 30 (test1) {'precision': 0.5219604682252553, 'recall': 0.7361859292236855, 'f1-score': 0.5728981285236443, 'average precision': 0.45115118659603126}        
    # epoch 50 (test2) {'precision': 0.6146919796160727, 'recall': 0.7494569581640756, 'f1-score': 0.6359045529747034, 'average precision': 0.5254145702800185, 'mean average precision': 0.2542652866757489}    
               
    
        
c = MLC(1)
if False :
    pass
    # c.train_MLC("1")
    # c.loadModel()
    # c.printSummaryNetwork( (3,224,224) )
    # c.test_MLC(load_results= True)
elif False:   
    pass
    # c.continue_training()
else:
    img = Image.open("./test2.jpg").convert('RGB')
    img_temp = np.array(img)
    plt.imshow(img_temp)
    plt.show()
    # c.model.to("cpu")
    c.forward(img, True)



    
