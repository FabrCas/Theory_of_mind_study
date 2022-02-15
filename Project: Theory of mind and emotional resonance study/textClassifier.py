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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multioutput import MultiOutputRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,explained_variance_score
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
        self.scaler_emb = None
        
        self.scaler_tc = None
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
    
    # check whether there's a word that has been wrongly recognized (is out of the context)
    
    def _filterRecognizedText(self,text, analyze_word):
        if not(self.name_embedding == "bert"): return 
        

        text = re.sub(r'[^\w\s]','',text) 
        
        words_list = word_tokenize(text)
        
        words_list = [self.stemmer.stem(word) for word in words_list]
        
        
        x_emb = self._wordToEmbedding(words_list, from_speech= True)

        analyze_word = 6
        
        x_avg = np.delete(x_emb, analyze_word, axis = 0)
        # print(x_avg.shape)
        average_vec = np.mean(x_avg, axis = 0)
        average_vec = average_vec.reshape(1,-1)
        # print(x_emb.shape)
        # print(average_vec.shape)
        
        similarities = []
        for i,emb_word in enumerate(x_emb):
            similarity = cosine_similarity(emb_word.reshape(1,-1), average_vec)

            similarities.append(similarity[0][0])
            
        print(similarities)
        

            
        # abs_max = np.max(np.abs(x_emb))
        
        # print(abs_max)
        
        # x_emb /= abs_max
        
        
        return x_emb
        
        
    
    def _preProcessSentence(self, sentence):
        sentence = sentence.lower()

        #remove punctuations and other non alphabetic characters
        sentence = re.sub(r'[^\w\s]','',sentence) 

        words_list = word_tokenize(sentence)

        # remove english stopwords
        words_list = [word for word in words_list if not word in self.stop_words]

        if self.useStemming:
            words_list = [self.stemmer.stem(word) for word in words_list]
        
        # todo: eliminate words not related to the sentences, i.e. error in the speech recognition phase
        
        return words_list
            
        
    def _wordToEmbedding(self, x, is_training = False, is_testing = False, from_speech = False):
        
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
                    self.scaler_emb = scaler
                
                x_emb_scaled = self.scaler_emb.transform(x_emb)
                
                print(x_emb_scaled.shape)   
                
            else: # simple forward
                x = x.strip()
    
                x_emb = np.array(self.embedding[x])
    
                x_emb = np.expand_dims(x_emb,0)
                x_emb_scaled = self.scaler_emb.transform(x_emb)
        
        elif self.name_embedding == "bert":
            
            if is_training or is_testing:
                
                # remove spaces from input
                x = [x_i.strip() for x_i in x] 
                
                # since here we work with words no sentence or sentences we can omit start & end tag: [CLS] [SEP]
                
                # tokenize word
                x_token = [self.tokenizer.tokenize(x_i) for x_i in x]
                
                
                # padding for different lenghts
                max_length_token = max([len(x_i) for x_i in x_token])
                for x_i in x_token:
                    while(len(x_i) < max_length_token):
                        x_i.append('[PAD]')
                
                # token to id
                x_token_idx = [self.tokenizer.convert_tokens_to_ids(x_i) for x_i in x_token]
                
                # token ids to Tensor
                x_token_idx = T.tensor(x_token_idx).to(self.device)
                
                with T.no_grad():
                    embedding_layers, _ = self.embedding(x_token_idx)
                
                # take embedding from the final layer (|h|=12)
                x_emb = embedding_layers[11].to('cpu').numpy()
                
                # take first element of the embedding
                x_emb = x_emb[:,0,:]

                if is_training:
                    # define scaler and scale input 
                    scaler = preprocessing.StandardScaler().fit(x_emb)
                    self.scaler_emb = scaler
                
                x_emb_scaled = self.scaler_emb.transform(x_emb)
                
            elif from_speech:
                #check dimension(multi/single phrase case)
                # if(type(x[0]) == str): #single sentence
                
                    # strip no more needed after the pre-processing
                    x_token = [self.tokenizer.tokenize(x_i) for x_i in x]
                    
                    max_length_token = max([len(x_i) for x_i in x_token])
                    for x_i in x_token:
                        while(len(x_i) < max_length_token):
                            x_i.append('[PAD]')
                            
                    print(x_token)
                            
                    x_token_idx = [self.tokenizer.convert_tokens_to_ids(x_i) for x_i in x_token]
                    
                    x_token_idx = T.tensor(x_token_idx).to(self.device)
                    
                    with T.no_grad():
                        embedding_layers, _ = self.embedding(x_token_idx)
                    
                    x_emb = embedding_layers[11].to('cpu').numpy()
                    
                    x_emb = x_emb[:,0,:]
                    
                    if self.scaler_emb == None:
                        self.loadScalerEmb()
                        
                    x_emb_scaled = self.scaler_emb.transform(x_emb)
                    
                # else: #multi sentences 
                #     print("multi sentences case")
                #     print(x)
                    
                #     x_token = [[self.tokenizer.tokenize(x_i) for x_i in x_word] for x_word in x]
                #     print(x_token)
                    
                #     max_length_token = max([max([len(x_i) for x_i in x_word]) for x_word in x_token])
                #     for x_word in x_token:
                #         for x_i in x_word:
                #             while(len(x_i) < max_length_token):
                #                 x_i.append('[PAD]')
                                
                #     print(x_token)
                    
                #     x_token_idx = [[self.tokenizer.convert_tokens_to_ids(x_i) for x_i in x_word] for x_word in x_token]
                #     print(x_token_idx)
                #     x_token_idx = T.tensor(x_token_idx).to(self.device)
                    
                    
            else: 
                # simple forward on word
                x = x.strip()
                
                x_token = self.tokenizer.tokenize(x)
                x_token_idx = self.tokenizer.convert_tokens_to_ids(x_token)
                x_token_idx = T.tensor([x_token_idx]).to(self.device)
     
                
                with T.no_grad():
                    embedding_layers, _ = self.embedding(x_token_idx)
                
                x_emb = embedding_layers[11].to('cpu').numpy()
                x_emb = x_emb[:,0,:]
                if self.scaler_emb is None:
                    self.loadScalerEmb()
                x_emb_scaled = self.scaler_emb.transform(x_emb)
            
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
      
    # ---------------------------- [load and save] ----------------------------
    
    def _saveModel(self, folder="models/TC_SVM/"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        if self.kernel_type == None:
            path_save = os.path.join(folder,"msvr.sav")
        else:
            path_save = os.path.join(folder,"msvr_"+ str(self.name_embedding)+ ".sav")

        pickle.dump(self.model, open(path_save, 'wb'))
    
    def loadModel(self, folder="models/TC_SVM/"):
        
        if self.kernel_type == None:
            path_load = os.path.join(folder,"msvr.sav")
        else:
            path_load = os.path.join(folder,"msvr_"+ str(self.name_embedding)+ ".sav")
        
        self.model = pickle.load(open(path_load,'rb'))
        self.loadScalerEmb()
        self.loadScalerTC()
    
    def _saveScalerEmb(self, folder="models/TC_SVM/scaler"):
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        path_save = os.path.join(folder,"scaler_xTrain_"+str(self.name_embedding) +".sav")
        pickle.dump(self.scaler_emb, open(path_save, 'wb'))
    
    def loadScalerEmb(self, folder="models/TC_SVM/scaler"):
        path_save = os.path.join(folder,"scaler_xTrain_"+str(self.name_embedding) +".sav")
        self.scaler_emb = pickle.load(open(path_save,'rb'))
        
    def _saveScalerTC(self, folder="models/TC_SVM/scaler"):
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        path_save = os.path.join(folder,"scaler_yTrain.sav")
        pickle.dump(self.scaler_emb, open(path_save, 'wb'))
    
    def loadScalerTC(self, folder="models/TC_SVM/scaler"):
        path_save = os.path.join(folder,"scaler_yTrain.sav")
        self.scaler_emb = pickle.load(open(path_save,'rb'))
    
    # ---------------------------- [load and save] ----------------------------

    def _computeMetrics(self,y_pred, y_target):
        
        # estimate the errors
        
        evaluations = {}
        
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
        
        
        evaluations['mse'] = mses
        evaluations['mae'] = maes
        evaluations['rmse'] = rmses
        
        
        variance_score = explained_variance_score(y_target, y_pred, multioutput="variance_weighted")
        print("-- Explained variance score -> {:.8f}".format(variance_score))
        
        r2 = r2_score(y_target, y_pred, multioutput="variance_weighted")
        print("-- R2 score-> {:.8f}".format(r2))
        
        return evaluations
    
    
    
    def train_TC(self, save_model = True):
        print("- Training the text classifier model...")
        #works fine glove
        # self.model = MultiOutputRegressor(svm.SVR(kernel= "rbf", degree=3 , \
        #                                           gamma = "scale", C =1000, epsilon= 1e-8, cache_size= 2000, max_iter= -1, tol = 1e-3))
        
        
        # self.model = MultiOutputRegressor(svm.SVR(kernel=self.kernel_type, degree=50 , \
        #                                           gamma = "scale", C =10, epsilon= 1e-10, cache_size= 2000, max_iter= 1e6, tol = 1e-8))
        
        # self.model = MultiOutputRegressor(svm.SVR(max_iter= -1) )   
        # self.model = MultiOutputRegressor(linear_model.SGDRegressor(max_iter=10000, tol=1e-10, loss="huber", epsilon= 1e-3, shuffle= True))
        # self.model = MultiOutputRegressor(svm.LinearSVR(max_iter=1000)) 
        
        # self.model = svm.SVR(kernel=self.kernel_type, degree= self.degree, \
        #                                           gamma = self.gamma, C = self.C, epsilon= self.eps)
        
        
        # --------------------------------------- start train test model tuning
        # self.model = MultiOutputRegressor(linear_model.SGDRegressor(max_iter=10000, tol=1e-10, loss="huber", epsilon= 1e-3, shuffle= True))
        # - Computing evaluation metrics for the text classifier:
        # -- Disgust:      MSE -> 0.68201780  RMSE -> 0.82584369  MAE -> 0.41854708
        # -- Surprise:     MSE -> 0.69519581  RMSE -> 0.83378403  MAE -> 0.49595749
        # -- Neutral:      MSE -> 0.64483460  RMSE -> 0.80301594  MAE -> 0.43231771
        # -- Anger:        MSE -> 0.74061623  RMSE -> 0.86059063  MAE -> 0.50340635
        # -- Sad:          MSE -> 0.70089675  RMSE -> 0.83719576  MAE -> 0.50026842
        # -- Happy:        MSE -> 0.71845831  RMSE -> 0.84761920  MAE -> 0.50647793
        # -- Fear:         MSE -> 0.73188894  RMSE -> 0.85550508  MAE -> 0.50660566
        # -- Global error: MSE -> 0.70198692  RMSE -> 0.83784660  MAE -> 0.48051152
        # -- Explained variance score -> 0.31908248
        # -- R2 score-> 0.29801308
        # - Testing the text classifier model...
        # - Computing evaluation metrics for the text classifier:
        # -- Disgust:      MSE -> 1.20105321  RMSE -> 1.09592573  MAE -> 0.72743288
        # -- Surprise:     MSE -> 1.27785960  RMSE -> 1.13042452  MAE -> 0.80453054
        # -- Neutral:      MSE -> 1.00422717  RMSE -> 1.00211136  MAE -> 0.70354187
        # -- Anger:        MSE -> 1.57838628  RMSE -> 1.25633844  MAE -> 0.83522595
        # -- Sad:          MSE -> 1.11618963  RMSE -> 1.05649876  MAE -> 0.78382820
        # -- Happy:        MSE -> 1.06317537  RMSE -> 1.03110396  MAE -> 0.78289954
        # -- Fear:         MSE -> 1.19423278  RMSE -> 1.09280958  MAE -> 0.79360946
        # -- Global error: MSE -> 1.20501772  RMSE -> 1.09773299  MAE -> 0.77586692
        # -- Explained variance score -> -0.11904400
        # -- R2 score-> -0.15021728
        # ----------------------------
        # self.model = MultiOutputRegressor(linear_model.SGDRegressor(max_iter=100000, tol=1e-10, loss="huber", epsilon= 1e-3, shuffle= True))
        # - Computing evaluation metrics for the text classifier:
        # -- Disgust:      MSE -> 0.65478648  RMSE -> 0.80918878  MAE -> 0.39696255
        # -- Surprise:     MSE -> 0.64283084  RMSE -> 0.80176732  MAE -> 0.45041795
        # -- Neutral:      MSE -> 0.56919010  RMSE -> 0.75444688  MAE -> 0.38879843
        # -- Anger:        MSE -> 0.69433832  RMSE -> 0.83326965  MAE -> 0.46996551
        # -- Sad:          MSE -> 0.62799363  RMSE -> 0.79246049  MAE -> 0.44676221
        # -- Happy:        MSE -> 0.66795894  RMSE -> 0.81728755  MAE -> 0.46307395
        # -- Fear:         MSE -> 0.70083705  RMSE -> 0.83716011  MAE -> 0.47897994
        # -- Global error: MSE -> 0.65113362  RMSE -> 0.80692851  MAE -> 0.44213722
        # -- Explained variance score -> 0.37042711
        # -- R2 score-> 0.34886638
        # - Testing the text classifier model...
        # - Computing evaluation metrics for the text classifier:
        # -- Disgust:      MSE -> 1.24676049  RMSE -> 1.11658429  MAE -> 0.74595522
        # -- Surprise:     MSE -> 1.39270070  RMSE -> 1.18012741  MAE -> 0.84818268
        # -- Neutral:      MSE -> 1.06426576  RMSE -> 1.03163257  MAE -> 0.74626431
        # -- Anger:        MSE -> 1.67676358  RMSE -> 1.29489906  MAE -> 0.86720580
        # -- Sad:          MSE -> 1.21756164  RMSE -> 1.10343176  MAE -> 0.83065284
        # -- Happy:        MSE -> 1.18422187  RMSE -> 1.08821959  MAE -> 0.83723900
        # -- Fear:         MSE -> 1.23932993  RMSE -> 1.11325196  MAE -> 0.82158852
        # -- Global error: MSE -> 1.28880057  RMSE -> 1.13525353  MAE -> 0.81386977
        # -- Explained variance score -> -0.20530776
        # -- R2 score-> -0.23018994
        # ----------------------------
        self.model = MultiOutputRegressor(svm.SVR(kernel= "rbf",\
                                                  gamma = "scale", C =100, epsilon= 1e-8, cache_size= 2000, max_iter= -1, tol = 1e-5))
        # - Computing evaluation metrics for the text classifier:
        # -- Disgust:      MSE -> 0.00000000  RMSE -> 0.00000232  MAE -> 0.00000191
        # -- Surprise:     MSE -> 0.00000000  RMSE -> 0.00000228  MAE -> 0.00000190
        # -- Neutral:      MSE -> 0.00000000  RMSE -> 0.00000222  MAE -> 0.00000185
        # -- Anger:        MSE -> 0.00000000  RMSE -> 0.00000221  MAE -> 0.00000182
        # -- Sad:          MSE -> 0.00000000  RMSE -> 0.00000230  MAE -> 0.00000190
        # -- Happy:        MSE -> 0.00000000  RMSE -> 0.00000232  MAE -> 0.00000194
        # -- Fear:         MSE -> 0.00000000  RMSE -> 0.00000222  MAE -> 0.00000185
        # -- Global error: MSE -> 0.00000000  RMSE -> 0.00000227  MAE -> 0.00000188
        # -- Explained variance score -> 1.00000000
        # -- R2 score-> 1.00000000
        # - Testing the text classifier model...
        # - Computing evaluation metrics for the text classifier:
        # -- Disgust:      MSE -> 1.05601337  RMSE -> 1.02762511  MAE -> 0.69402262
        # -- Surprise:     MSE -> 1.13217727  RMSE -> 1.06403819  MAE -> 0.75169798
        # -- Neutral:      MSE -> 0.85193311  RMSE -> 0.92300223  MAE -> 0.68601744
        # -- Anger:        MSE -> 1.52073789  RMSE -> 1.23318202  MAE -> 0.83028540
        # -- Sad:          MSE -> 0.99861746  RMSE -> 0.99930849  MAE -> 0.74082661
        # -- Happy:        MSE -> 0.91396671  RMSE -> 0.95601606  MAE -> 0.70172916
        # -- Fear:         MSE -> 1.21846705  RMSE -> 1.10384195  MAE -> 0.82403264
        # -- Global error: MSE -> 1.09884470  RMSE -> 1.04825793  MAE -> 0.74694455
        # -- Explained variance score -> -0.04263489
        # -- R2 score-> -0.04887267
        # ----------------------------
        # self.model = MultiOutputRegressor(svm.SVR(kernel= "poly", degree=15 , \
        #                                           gamma = "scale", C =1, epsilon= 1e-8, cache_size= 2000, max_iter= 10000, tol = 1e-20))
        # - Computing evaluation metrics for the text classifier:
        # -- Disgust:      MSE -> 1.20414182  RMSE -> 1.09733396  MAE -> 0.67794152
        # -- Surprise:     MSE -> 1.08768902  RMSE -> 1.04292330  MAE -> 0.70928672
        # -- Neutral:      MSE -> 1.08449716  RMSE -> 1.04139194  MAE -> 0.70862276
        # -- Anger:        MSE -> 1.58653129  RMSE -> 1.25957584  MAE -> 0.79186453
        # -- Sad:          MSE -> 0.96416530  RMSE -> 0.98191919  MAE -> 0.69995105
        # -- Happy:        MSE -> 0.88559179  RMSE -> 0.94105887  MAE -> 0.65996352
        # -- Fear:         MSE -> 1.10232244  RMSE -> 1.04991544  MAE -> 0.72696950
        # -- Global error: MSE -> 1.13070555  RMSE -> 1.06334639  MAE -> 0.71065709
        # -- Explained variance score -> -0.00410389
        # -- R2 score-> -0.07928459
        # - Computing evaluation metrics for the text classifier:
        # -- Disgust:      MSE -> 0.67043601  RMSE -> 0.81880157  MAE -> 0.37866740
        # -- Surprise:     MSE -> 0.63613557  RMSE -> 0.79758107  MAE -> 0.42149205
        # -- Neutral:      MSE -> 0.67979628  RMSE -> 0.82449759  MAE -> 0.40463515
        # -- Anger:        MSE -> 0.52126619  RMSE -> 0.72198767  MAE -> 0.38463306
        # -- Sad:          MSE -> 0.65961440  RMSE -> 0.81216648  MAE -> 0.42609316
        # -- Happy:        MSE -> 0.58094241  RMSE -> 0.76219578  MAE -> 0.40637442
        # -- Fear:         MSE -> 0.64181690  RMSE -> 0.80113476  MAE -> 0.41652845
        # -- Global error: MSE -> 0.62714396  RMSE -> 0.79192422  MAE -> 0.40548910
        # -- Explained variance score -> 0.39701860
        # -- R2 score-> 0.37285604
        
        #----------------------------
        #----------------------------
        #----------------------------
        #----------------------------
        #----------------------------
        #----------------------------
        #----------------------------
        #----------------------------
        #----------------------------
        # ----------------------------------------- end train test model tuning
        
        
        
        trainset = self._getTrainSet()
        
        # separate x and y from trainSet
        x,y = trainset[:,0],trainset[:,1:]
        
        # targets to numpy array 
        y = np.array(y)
            
        # scale targets
        self.scaler_tc = preprocessing.StandardScaler().fit(y)
        y = self.scaler_tc.transform(y)
        

        x_emb_scaled = self._wordToEmbedding(x, is_training= True)
        
        self.model.fit(x_emb_scaled,y)
        
        if save_model: 
            self._saveModel()
            self._saveScalerEmb()
            self._saveScalerTC()
            
        gc.collect()
        
    
    def test_TC(self, setType = "test"):
        print("- Testing the text classifier model...")
        
        if self.model == None:
            self.loadModel()
            
        if self.scaler_tc == None:
            self.loadScalerTC()
            
        if setType == "train": testset = self._getTrainSet()
        else: testset = self._getTestSet()
        
        # separate x and y from testSet
        x,y = testset[:,0],testset[:,1:]
        
        # targets to numpy array 
        y = np.array(y)
        
        # scale targets
        y = self.scaler_tc.transform(y)
        
        # process x
        x_emb_scaled = self._wordToEmbedding(x,is_testing=True)
        
        # make predictions
        y_pred = self.model.predict(x_emb_scaled)
        
        # print(y_pred.shape)
        # print(y_pred)
        # print(y.shape)
        # print(y)
        
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
        words = self._preProcessSentence(words)
        print(len(words))
        print(words)
        x_emb = self._wordToEmbedding(words, from_speech= True)
        print(x_emb.shape)
        y = self.model.predict(x_emb)
        print(y.shape)
        
        y = np.mean(y, axis=0)
        
        return y
        
        
    def predict_Corpus2Sentiment(self,c):
        
        if not(self.usePunctuator):
            # corpus as a full sentence
            return self.predict_wordsList2Sentiment(c)
        
        sents = self._separeteSentences(text)
        
        if len(sents)==1:
            # no sentences separation needed detected
            return self.predict_wordsList2Sentiment(c)
        
        y_sents = [self.predict_wordsList2Sentiment(words) for words in sents]
        print(y_sents)
        y = np.mean(y_sents, axis=0)
        print(y)
        
        return y

    def get_rankedEmotions(self, y_score):
        emotions = self.emo_sensor_reader.getLabels()
        ranked_emo = {}
        for idx,y_val in enumerate(y_score):
            ranked_emo[emotions[idx]] = y_score[idx]
        
        print(ranked_emo)
        ranked_emo = sorted(ranked_emo.items(), key = lambda kv:(kv[1], kv[0]), reverse = True )
        print(ranked_emo)
        return ranked_emo
    
    # generic forward method for word/s and sentence/s
    def forward(self, x):
        
        startTime = time.time()
        
        y = self.predict_Corpus2Sentiment(x)
        y_ranked = self.get_rankedEmotions(y)
        
        print("End prediction of emotions, time: {} [s]".format((time.time() -startTime)))
        return y,y_ranked
        
        

# ---------------------------> [test section]
        
new = TC(0)

# test sentence from asr emotions ranking
if False:
    new.loadModel()
    new.usePunctuator = True 
    text = "today i was having fun playing with my cousin when a stranger came up into the house he was tall and thin he asked about his parents but they weren't at home he said to let them know about the visit "
    new.get_rankedEmotions(new.predict_Corpus2Sentiment(text))


# test identification of wrong word recognized 
if False:
    
    # sentence_ok = "how are you? I am fine thanks"
    sentence_notok = "how are you? I pen thanks"
    
    new.usePunctuator = False
    # sentence_notok = "today i've played videogames all the day, so relaxing"
    analyze_word = 4
    
    emb = new._filterRecognizedText(sentence_notok,analyze_word)
    
        # analyze_word = 6 
        
        # print(words_list[analyze_word])
    
        # x_emb = self._wordToEmbedding(words_list, from_speech= True)
        
        # for i in range(len(words_list)):
        # -------------------------------------------------------------------------------------------- 
        #     x_avg = np.delete(x_emb, i, axis = 0)
        #     # print(x_avg.shape)
        #     average_vec = np.mean(x_avg, axis = 0)
        #     average_vec = average_vec.reshape(1,-1)
        #     # print(x_emb.shape)
        #     # print(average_vec.shape)
            
        #     similarities = []
        #     for i,emb_word in enumerate(x_emb):
        #         similarity = cosine_similarity(emb_word.reshape(1,-1), average_vec)

        #         similarities.append(similarity[0][0])
                
        #     print(similarities)
        
        # -------------------------------------------------------------------------------------------
            # model = BertForMaskedLM.from_pretrained('bert-base-uncased')
            # model.eval()
            
            # model.to(self.device)
            
            # text = re.sub(r'[^\w\s]','',text) 
            
            # words_list = word_tokenize(text)
            
            # words_list = [self.stemmer.stem(word) for word in words_list]
            
            # for word in words_list:
            #     print(word)
            
            # # look at speech case in word embeeding 
            
            # # for i in range(len(words_list)):
            # analyze_word = 6
                
            # words_list[analyze_word] = '[MASK]'
              
            # x_token = [self.tokenizer.tokenize(x_i) for x_i in words_list]
            
            # max_length_token = max([len(x_i) for x_i in x_token])
            # for x_i in x_token:
            #     while(len(x_i) < max_length_token):
            #         x_i.append('[PAD]')
                    
            # print(x_token)
                    
            # x_token_idx = [self.tokenizer.convert_tokens_to_ids(x_i) for x_i in x_token]
            
            # x_token_idx = T.tensor(x_token_idx).to(self.device)
            
            # with T.no_grad():
            #     predictions = model(x_token_idx)
            
            # predicted_index = T.argmax(predictions[0, analyze_word]).item()
            # predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])[0]
            
            # print(predicted_token)
                   
            # ------------------------------------------------------------------------------------------- 
            
    # print(emb.shape)

    # emb_ww = np.delete (emb,analyze_word, axis = 0)
    # print(emb_ww.shape)
    
    # average_vec = np.mean(emb_ww, axis = 0)

    

    # acc = 0
    # print("----------cos----------")
    # for i,emb_word in enumerate(emb):
    #     # print(emb_word.shape)
    #     cs = T.cosine_similarity( T.tensor( np.expand_dims(emb[analyze_word],axis= 0)),T.tensor( np.expand_dims(emb_word,axis= 0) ) )
    #     print(cs)
    #     acc += float(cs[0])
    
    # print(acc)
    # print("--------------------")
      
    # for emb_word in emb:
    #     # print(emb_word.shape)
    #     print(T.cosine_similarity( T.tensor( np.expand_dims(average_vec,axis= 0)),T.tensor( np.expand_dims(emb_word,axis= 0) ) ))
     

    # delta_vec = emb[analyze_word] - average_vec 
    # print(delta_vec.shape)
    # delta = np.linalg.norm(delta_vec)
    # print(delta)
    
    

# test single word prediction score
if True:
    new.train_TC(save_model=False)

    # t1 = "sun"
    # t2 = "injured"
    
    # y1,x1 = new.predict_Word2Sentiment(t1)
    # y2,x2 = new.predict_Word2Sentiment(t2)
    
    # print(T.cosine_similarity(T.tensor(x1),T.tensor(x2)))
    # print(T.cosine_similarity(T.tensor(y1),T.tensor(y2)))
    
    new.test_TC(setType ="train")
    new.test_TC(setType = "test")