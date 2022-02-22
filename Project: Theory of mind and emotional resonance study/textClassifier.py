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
    
    
    
    def train_TC(self, save_model = True, use_full = False):
        print("- Training the text classifier model...")
        
        self.model = MultiOutputRegressor(svm.SVR(kernel= "rbf",\
                                                  gamma = "scale", C = 5, epsilon= 1e-4, cache_size= 2000, max_iter= -1, tol = 1e-5))
        # - Training the text classifier model...
        # - Using the full dataset for training...
        # - Testing the text classifier model...
        # - Computing evaluation metrics for the text classifier:
        # -- Disgust:      MSE -> 0.10972640  RMSE -> 0.33124975  MAE -> 0.04245848
        # -- Surprise:     MSE -> 0.04602675  RMSE -> 0.21453846  MAE -> 0.03793087
        # -- Neutral:      MSE -> 0.03004032  RMSE -> 0.17332145  MAE -> 0.03118618
        # -- Anger:        MSE -> 0.04954111  RMSE -> 0.22257831  MAE -> 0.03512609
        # -- Sad:          MSE -> 0.05785598  RMSE -> 0.24053270  MAE -> 0.04432290
        # -- Happy:        MSE -> 0.04829081  RMSE -> 0.21975169  MAE -> 0.03961154
        # -- Fear:         MSE -> 0.06316022  RMSE -> 0.25131698  MAE -> 0.04356229
        # -- Global error: MSE -> 0.05780594  RMSE -> 0.24042866  MAE -> 0.03917119
        # -- Explained variance score -> 0.94296737
        # -- R2 score-> 0.94167403
        # - Testing the text classifier model...
        # - Computing evaluation metrics for the text classifier:
        # -- Disgust:      MSE -> 0.69622562  RMSE -> 0.83440135  MAE -> 0.52792654
        # -- Surprise:     MSE -> 0.70038943  RMSE -> 0.83689272  MAE -> 0.55796328
        # -- Neutral:      MSE -> 0.49501726  RMSE -> 0.70357463  MAE -> 0.52640148
        # -- Anger:        MSE -> 0.91329972  RMSE -> 0.95566716  MAE -> 0.63889260
        # -- Sad:          MSE -> 0.62127248  RMSE -> 0.78820840  MAE -> 0.56550408
        # -- Happy:        MSE -> 0.56738550  RMSE -> 0.75324996  MAE -> 0.54563259
        # -- Fear:         MSE -> 0.65362400  RMSE -> 0.80847016  MAE -> 0.60561190
        # -- Global error: MSE -> 0.66388772  RMSE -> 0.81479305  MAE -> 0.56684749
        # -- Explained variance score -> 0.36242966
        # -- R2 score-> 0.35639843
        # ----------------------------------------- end train test model tuning
        
        
        
        trainset = self._getTrainSet()
        
        if use_full:
            print("- Using the full dataset for training...")
            tmp = self._getTestSet()
            trainset = np.concatenate((trainset,tmp))
 
        
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
    new.train_TC(save_model=True, use_full=True)


    new.test_TC(setType ="train")
    new.test_TC(setType = "test")
    
    # t1 = "sun"
    # t2 = "injured"
    
    # y1,x1 = new.predict_Word2Sentiment(t1)
    # y2,x2 = new.predict_Word2Sentiment(t2)
    
    # print(T.cosine_similarity(T.tensor(x1),T.tensor(x2)))
    # print(T.cosine_similarity(T.tensor(y1),T.tensor(y2)))