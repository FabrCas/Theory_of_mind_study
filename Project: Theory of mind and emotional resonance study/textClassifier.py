# General imports 
import numpy as np 
import sys 
import time 
import os
import re
import math
import gc
import pickle

# local imports
from emotionDatasetAnalyzer import emotionSensorReader

# Machine learning libs 
import torchtext as TT
import torch as T
from sklearn import svm
from sklearn import linear_model
from sklearn.multioutput import MultiOutputRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from punctuator import Punctuator
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

class TC():
    def __init__(self, args):
        self.emo_sensor_reader = emotionSensorReader()
        # self.trainSet, self.testSet = self.emo_sensor_reader.loadSplittedDataset()
        
        self.name_embedding = "bert"   # to be selected from args
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.embedding = None
        self.tokenizer = None
        self.scaler = None
        self.model = None
        
        self.C = 1
        self.eps = 0.1  # eps-tube size for no penalty (squared l2 penalty)
        self.gamma = 'scale' #1e-8 #'scale'   # "auto" or "scale"(not used for linear kernel)
        self.degree = 3  # just for polynomial kernel
        self.kernel_type = 'poly'  # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’
        
        # load embedding
        self._loadEmbedding()
        
        self.usePunctuator = True # from args
        self.useStemming = True
        
        
        if self.usePunctuator:
            self.punctuator = Punctuator('./models/punctuator/Demo-Europarl-EN.pcl')
        
        if self.useStemming:
            self.stemmer = PorterStemmer()
        try:   
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
    
    # this operation increase accuracy but make computation slower
    def _separeteSentences(self, text):
        text = self.punctuator.punctuate(text)
        sents_list = sent_tokenize(text)
        return sents_list
    
    def _preProcessSentence(self, sentence):
        sentence = sentence.lower()

        #remove punctuations and other non alphabetic characters
        sentence = re.sub(r'[^\w\s]','',sentence) 

        words_list = word_tokenize(sentence)

        # remove english stopwords
        words_list = [word for word in words_list if not word in self.stop_words]

        if self.useStemming:
            words_list = [self.stemmer.stem(word) for word in words_list]

        return words_list
            
        
    def _wordToEmbedding(self, x, is_training = False, is_testing = False):
        
        if self.name_embedding == "glove":
            if is_training or is_testing:
                
                # remove spaces from input
                x = [x_i.strip() for x_i in x] 
                
                # get embedding
                x_emb = [np.array(self.embedding[x_i])for x_i in x]
                
    
                # to numpy array embeddings
                x_emb = np.array(x_emb)
                
    
                if is_training:
                    # define scaler and scale input 
                    scaler = preprocessing.StandardScaler().fit(x_emb)
                    self.scaler = scaler
                
                x_emb_scaled = self.scaler.transform(x_emb)
                
                print(x_emb_scaled.shape)   
                
            else: # simple forward
                x = x.strip()
    
                x_emb = np.array(self.embedding[x])
    
                x_emb = np.expand_dims(x_emb,0)
                x_emb_scaled = self.scaler.transform(x_emb)
        
        elif self.name_embedding == "bert":
            # since here we work with words no sentence or sentences we can omit start & end tag: [CLS] [SEP]
     
            
            if is_training or is_testing:
                
                # remove spaces from input
                x = [x_i.strip() for x_i in x] 
                
                # print(x)
                # tokenize word
                x_token = [self.tokenizer.tokenize(x_i) for x_i in x]
                
                
                # padding for different lenghts
                max_length_token = max([len(x_i) for x_i in x_token])
                for x_i in x_token:
                    while(len(x_i) < max_length_token):
                        x_i.append('[PAD]')
                # print(x_token)
                
                # token to id
                x_token_idx = [self.tokenizer.convert_tokens_to_ids(x_i) for x_i in x_token]
                # print(x_token_idx)
                
                # token ids to Tensor
                x_token_idx = T.tensor(x_token_idx).to(self.device)
                # print(x_token_idx.shape)
                
                with T.no_grad():
                    embedding_layers, _ = self.embedding(x_token_idx)
                
                # take embedding from the final layer (|h|=12)
                x_emb = embedding_layers[11].to('cpu').numpy()
                
                # print(x_emb.shape)
                # take first element of the embedding
                x_emb = x_emb[:,0,:]
                # print(x_emb[3])
                # print(x_emb.shape)
                
            
                if is_training:
                    # define scaler and scale input 
                    scaler = preprocessing.StandardScaler().fit(x_emb)
                    self.scaler = scaler
                
                x_emb_scaled = self.scaler.transform(x_emb)
                
                # print(x_emb_scaled[3])
                # print(x_emb_scaled.shape)   
                
            elif not(isinstance(x, list)) : # simple forward on word
                x = x.strip()
                
                x_token = self.tokenizer.tokenize(x)
                x_token_idx = self.tokenizer.convert_tokens_to_ids(x_token)
                x_token_idx = T.tensor([x_token_idx]).to(self.device)
 
                
                with T.no_grad():
                    embedding_layers, _ = self.embedding(x_token_idx)
                
                x_emb = embedding_layers[11].to('cpu').numpy()
                x_emb = x_emb[:,0,:]
                x_emb_scaled = self.scaler.transform(x_emb)
                
            else: # forward
                pass
            
        return x_emb_scaled
            
        
        
    def _loadEmbedding(self):
            # glove = TT.vocab.GloVe(name="6B", dim=100)
            
            if self.name_embedding == "glove":
                self.embedding = TT.vocab.GloVe(name="840B", dim=300)
                
            elif self.name_embedding == "bert":
                self.embedding = BertModel.from_pretrained('bert-base-uncased')
                self.embedding.eval()
                self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.embedding.to(self.device)
            
            
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
            path_save = os.path.join(folder,"msvr_"+ str(self.name_embedding)+ ".sav")
        # print(path_save)
        pickle.dump(self.model, open(path_save, 'wb'))
    
    def _saveScaler(self, folder="models/TC_SVM/scaler"):
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        path_save = os.path.join(folder,"scaler_xTrain_"+str(self.name_embedding) +".sav")
        pickle.dump(self.scaler, open(path_save, 'wb'))
    
    def loadScaler(self, folder="models/TC_SVM/scaler"):
        path_save = os.path.join(folder,"scaler_xTrain_"+str(self.name_embedding) +".sav")
        self.scaler = pickle.load(open(path_save,'rb'))
        
    
    def loadModel(self, folder="models/TC_SVM/"):
        
        if self.kernel_type == None:
            path_load = os.path.join(folder,"msvr.sav")
        else:
            path_load = os.path.join(folder,"msvr_"+ str(self.name_embedding)+ ".sav")
        
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
    
    
    
    def train_TC(self, save_model = True):
        print("- Training regresssion model...")
        
        #works fine glove
        self.model = MultiOutputRegressor(svm.SVR(kernel= "rbf", degree=3 , \
                                                  gamma = "scale", C =1000, epsilon= 1e-8, cache_size= 2000, max_iter= -1, tol = 1e-3))
        
        
        # self.model = MultiOutputRegressor(svm.SVR(kernel=self.kernel_type, degree=50 , \
        #                                           gamma = "scale", C =10, epsilon= 1e-10, cache_size= 2000, max_iter= 1e6, tol = 1e-8))
        
        # self.model = MultiOutputRegressor(svm.SVR(max_iter= -1) )   
        # self.model = MultiOutputRegressor(linear_model.SGDRegressor(max_iter=100000, tol=1e-3))
        # self.model = MultiOutputRegressor(svm.LinearSVR(max_iter=10000)) 
        
        
        # self.model = svm.SVR(kernel=self.kernel_type, degree= self.degree, \
        #                                           gamma = self.gamma, C = self.C, epsilon= self.eps)
        
        trainset = self._getTrainSet()
        
        # separate x and y from trainSet
        x,y = trainset[:,0],trainset[:,1:]
        
        # targets to numpy array 
        y = np.array(y)
        
        x_emb_scaled = self._wordToEmbedding(x, is_training= True)
        
        self.model.fit(x_emb_scaled,y)
        
        if save_model: 
            self._saveModel()
            self._saveScaler()
            
        gc.collect()
        
    
    def test_TC(self):
        print("- Testing regresssion model...")
        
        if self.model == None:
            self.loadModel()
            
        testset = self._getTestSet()
        # separate x and y from testSet
        x,y = testset[:,0],testset[:,1:]
        
        # targets to numpy array 
        y = np.array(y)
        
        # process x
        x_emb_scaled = self._wordToEmbedding(x,is_testing=True)
        
        # make predictions
        y_pred = self.model.predict(x_emb_scaled)
        
        # measure the error
        self._computeMetrics(y_pred,y)
        gc.collect()
        
    def predict_Word2Sentiment(self, x):
        
        if self.model == None:
            self.loadModel()
        
        # process x
        x_emb = self._wordToEmbedding(x)
        
        # make predictions
        y = self.model.predict(x_emb)
        print(y.shape)
        
        print(y)
        return y, x_emb
    
    def predict_wordsList2Sentiment(self, words):
        print(words)
        
    def predict_Corpus2Sentiment(self,c):
        if not(self.usePunctuator):
            return self.predict_wordsList2Sentiment(c)
        
        sents = self._separeteSentences(text)
        sents = [self._preProcessSentence(sent) for sent in sents]
        y_sents = [self.predict_wordsList2Sentiment(words) for words in sents]
    

        
        
new = TC(0)
text = "today i was having fun playing with my cousin when a stranger came up into the house he was tall and thin he asked about his parents but they weren't at home he said to let them know about the visit "
new.predict_Corpus2Sentiment(text)

if False:
    new.train_TC()
    
    # new.loadModel()
    # new.test_TC()
    
    t1 = "sun"
    t2 = "injured"
    
    y1,x1 = new.predict_Word2Sentiment(t1)
    y2,x2 = new.predict_Word2Sentiment(t2)
    print(T.cosine_similarity(T.tensor(x1),T.tensor(x2)))
    print(T.cosine_similarity(T.tensor(y1),T.tensor(y2)))