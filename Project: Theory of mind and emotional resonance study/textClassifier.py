import numpy as np 
import sys 
import time 
import os
import math
import gc
import torch as T
from emotionDatasetAnalyzer import emotionSensorReader
import torchtext as TT
from sklearn import svm
from sklearn import linear_model
from sklearn.multioutput import MultiOutputRegressor
from sklearn import preprocessing
import pickle
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error





device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

class TC():
    def __init__(self, args):
        self.emo_sensor_reader = emotionSensorReader()
        # self.trainSet, self.testSet = self.emo_sensor_reader.loadSplittedDataset()
        
        self.name_embedding = "glove"   # to be selected from args
        self.embedding = None
        self.tokenizer = None
        self.scaler = None
        self.model = None
        
        self.C = 1
        self.eps = 0.1  # eps-tube size for no penalty (squared l2 penalty)
        self.gamma = 'scale' #1e-8 #'scale'   # "auto" or "scale"(not used for linear kernel)
        self.degree = 3  # just for polynomial kernel
        self.kernel_type = 'rbf'  # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
        
        # load embedding
        self._embedding()
        
        
        
        
        
    def _embedding(self):
            # glove = TT.vocab.GloVe(name="6B", dim=100)
            
            if self.name_embedding == "glove":
                self.embedding = TT.vocab.GloVe(name="840B", dim=300)
            else:
                pass
            
            # t1 =  glove['cat'].unsqueeze(0)
            # t2 =   glove['dog'].unsqueeze(0)
            
            # print(T.cosine_similarity(t1,t2))
            
    def _getTestSet(self):
        _ , testSet = self.emo_sensor_reader.loadSplittedDataset()
        return testSet
        
    def _getTrainSet(self):
        trainSet, _ = self.emo_sensor_reader.loadSplittedDataset()
        return trainSet
        
    def _saveModel(self, folder="models/TC_SVM/"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        if self.kernel_type == None:
            path_save = os.path.join(folder,"msvr.sav")
        else:
            path_save = os.path.join(folder,"msvr_"+ str(self.kernel_type)+ ".sav")
        # print(path_save)
        pickle.dump(self.model, open(path_save, 'wb'))
    
    def _saveScaler(self, folder="models/TC_SVM/scaler"):
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        path_save = os.path.join(folder,"scaler_xTrain.sav")
        pickle.dump(self.scaler, open(path_save, 'wb'))
    
    def loadScaler(self, folder="models/TC_SVM/scaler"):
        path_save = os.path.join(folder,"scaler_xTrain.sav")
        self.scaler = pickle.load(open(path_save,'rb'))
        
    
    def loadModel(self, folder="models/TC_SVM/"):
        
        if self.kernel_type == None:
            path_load = os.path.join(folder,"msvr.sav")
        else:
            path_load = os.path.join(folder,"msvr_"+ str(self.kernel_type)+ ".sav")
        
        self.model = pickle.load(open(path_load,'rb'))
        self.loadScaler()
    
    def _computeMetrics(self,y_pred, y_target):
        
        mses = []
        maes = []
        rmses = []
        
        print("- Computing evaluation metrics for the text classifier:")
        
        mse_disgust = mean_squared_error(y_target[:,0], y_pred[:,0])
        mae_disgust = mean_absolute_error(y_target[:,0], y_pred[:,0])
        rmse_disgust = math.sqrt(mse_disgust)
        mses.append(mse_disgust);maes.append(mae_disgust),rmses.append(rmse_disgust)
        print("-- Disgust:      MSE -> {:.8f}  RMSE -> {:.8f}  MAE -> {:.8f}".format(mse_disgust, rmse_disgust, mae_disgust))
        
        mse_surprise = mean_squared_error(y_target[:,1], y_pred[:,1])
        mae_surprise= mean_absolute_error(y_target[:,1], y_pred[:,1])
        rmse_surprise = math.sqrt(mse_surprise)
        mses.append(mse_surprise);maes.append(mae_surprise);rmses.append(rmse_surprise)
        print("-- Surprise:     MSE -> {:.8f}  RMSE -> {:.8f}  MAE -> {:.8f}".format(mse_surprise, rmse_surprise, mae_surprise))
        
        mse_neutral = mean_squared_error(y_target[:,2], y_pred[:,2])
        mae_neutral= mean_absolute_error(y_target[:,2], y_pred[:,2])
        rmse_neutral = math.sqrt(mse_neutral)
        mses.append(mse_neutral);maes.append(mae_neutral);rmses.append(rmse_neutral)
        print("-- Neutral:      MSE -> {:.8f}  RMSE -> {:.8f}  MAE -> {:.8f}".format(mse_neutral, rmse_neutral, mae_neutral))
        
        mse_anger = mean_squared_error(y_target[:,3], y_pred[:,3])
        mae_anger = mean_absolute_error(y_target[:,3], y_pred[:,3])
        rmse_anger = math.sqrt(mse_anger)
        mses.append(mse_anger);maes.append(mae_anger);rmses.append(rmse_anger)
        print("-- Anger:        MSE -> {:.8f}  RMSE -> {:.8f}  MAE -> {:.8f}".format(mse_anger, rmse_anger, mae_anger))
        
        mse_sad = mean_squared_error(y_target[:,4], y_pred[:,4])
        mae_sad = mean_absolute_error(y_target[:,4], y_pred[:,4])
        rmse_sad = math.sqrt(mse_sad)
        mses.append(mse_sad);maes.append(mae_sad);rmses.append(rmse_sad)
        print("-- Sad:          MSE -> {:.8f}  RMSE -> {:.8f}  MAE -> {:.8f}".format(mse_sad, rmse_sad, mae_sad))
        
        mse_happy = mean_squared_error(y_target[:,5], y_pred[:,5])
        mae_happy = mean_absolute_error(y_target[:,5], y_pred[:,5])
        rmse_happy = math.sqrt(mse_happy)
        mses.append(mse_happy);maes.append(mae_happy);rmses.append(rmse_happy)
        print("-- Happy:        MSE -> {:.8f}  RMSE -> {:.8f}  MAE -> {:.8f}".format(mse_happy,rmse_happy, mae_happy))
        
        mse_fear = mean_squared_error(y_target[:,6], y_pred[:,6])
        mae_fear = mean_absolute_error(y_target[:,6], y_pred[:,6])
        rmse_fear = math.sqrt(mse_fear)
        mses.append(mse_fear);maes.append(mae_fear);rmses.append(rmse_fear)
        print("-- Fear:         MSE -> {:.8f}  RMSE -> {:.8f}  MAE -> {:.8f}".format(mse_fear, rmse_fear, mae_fear))
        
        mse_global = mean_squared_error(y_target, y_pred)
        mae_global = mean_absolute_error(y_target, y_pred)
        rmse_global= math.sqrt(mse_global)
        mses.append(mse_global);maes.append(mae_global);rmses.append(rmse_global)
        print("-- Global error: MSE -> {:.8f}  RMSE -> {:.8f}  MAE -> {:.8f}".format(mse_global, rmse_global, mae_global) )

        return mses, maes, rmses
    
    
    
    def train_TC(self, save_model = False, kernel_type = None):
        
        # print(self.kernel_type)
        self.model = MultiOutputRegressor(svm.SVR(kernel=self.kernel_type, degree=3 , \
                                                  gamma = self.gamma, C =1000, epsilon= 1e-8, cache_size= 2000, max_iter= -1))
        # self.model = MultiOutputRegressor(svm.SVR(max_iter= -1) )   
        # self.model = MultiOutputRegressor(linear_model.SGDRegressor(max_iter=1e9, tol=1e-3))
        # self.model = MultiOutputRegressor(svm.LinearSVR()) 
        
        
        # self.model = svm.SVR(kernel=self.kernel_type, degree= self.degree, \
        #                                           gamma = self.gamma, C = self.C, epsilon= self.eps)
        
        trainset = self._getTrainSet()
        
        # print(trainset)
        print(trainset.shape)
        # separate x and y from trainSet
        x,y = trainset[:,0],trainset[:,1:]
        
        # print(x)
        print(x.shape)
        
        # remove spaces from input
        x = [x_i.strip() for x_i in x] 
        
        # get embedding
        x_emb = [np.array(self.embedding[x_i])for x_i in x]
        

        
        # to numpy array
        x_emb = np.array(x_emb)
        y = np.array(y)
        
        # print(x_emb)
        print(x_emb.shape)
        
        # print(y)
        print(y.shape)
        
        # define scaler and scale input 
        scaler = preprocessing.StandardScaler().fit(x_emb)
        self.scaler = scaler
        x_emb_scaled = scaler.transform(x_emb)
        
        # print(x_emb_scaled)
        print(x_emb_scaled.shape)

        
        # print(x_emb_scaled.shape)
        # print(y.shape)
        
        # x_emb_scaled = x_emb_scaled[:,:1]
        # y = y[:,:]
        
        
        print(x_emb_scaled.shape)
        self.model.fit(x_emb_scaled,y)
        t1 = "happy"
        t2 = "die"
        
        print(T.cosine_similarity(self.embedding[t1].unsqueeze(0),self.embedding[t2].unsqueeze(0)))
        
        print("------------------------------------------------------")
        t1 = t1.strip()
        t1 = np.array(self.embedding[t1])
        # t1 = np.squeeze(t1)
        t1 = np.expand_dims(t1,0)
        
        t2 = t2.strip()
        t2 = np.array(self.embedding[t2])
        # t2 = np.squeeze(t2)
        t2 = np.expand_dims(t2,0)
        
        y1 = self.model.predict(t1)
        y2 = self.model.predict(t2)
        print(y1)
        print(y2)
        print(T.cosine_similarity(T.tensor(y1),T.tensor(y2)))
        
        if save_model: 
            self._saveModel()
            self._saveScaler()
            
        gc.collect()
        
    
    def test_TC(self):
        testset = self._getTestSet()
        # separate x and y from testSet
        x,y = testset[:,0],testset[:,1:]
        
        # process x
        x = [x_i.strip() for x_i in x] 
        x_emb = [np.array(self.embedding[x_i])for x_i in x]
        y = np.array(y)
        x_emb_scaled = self.scaler.transform(x_emb)
        
        # make predictions
        y_pred = self.model.predict(x_emb_scaled)
        
        # measure the error
        self._computeMetrics(y_pred,y)
        gc.collect()
        
    def predict(self, x):
        x = x.strip()
        # x = np.expand_dims(x,0)
        x = np.array(self.embedding[x])
        # print(x)
        print(x.shape)
        x = np.expand_dims(x,0)
        # x = self.scaler.transform(x)

        print(x.shape)
        y = self.model.predict(x)
        print(y.shape)
        return y
        
        
        
        
new = TC(0)
new.train_TC()
# new.loadModel()
# new.test_TC()

t1 = "happy"
t2 = "die"
# print(new.predict(t1))
# print(new.predict(t2))

# print(T.cosine_similarity(new.embedding[t1].unsqueeze(0),new.embedding[t2].unsqueeze(0)))
