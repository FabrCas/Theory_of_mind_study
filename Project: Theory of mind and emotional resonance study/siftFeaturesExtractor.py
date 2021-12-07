import cv2
import numpy as np
import os
import pandas as pd
import csv
from random import randrange

import matplotlib.pyplot as plt
from scipy.sparse.construct import random

from sklearn.cluster import MiniBatchKMeans
from sklearn.neural_network import MLPClassifier

#img_path = 'dataset/nus-wide/images'
image_test = './dataset/nus-wide/images/86954_1464792520_f45dec2e92_m.jpg'
#image_test = "./test.jpg"
fixed_n_classes = 81
#train = pd.read_csv('../input/train.csv')
#species = train.species.sort_values().unique()

files = os.listdir("./dataset/nus-wide/images/")
index = randrange(0,len(files))
print(index)

#todo 0

#for filename in files:
#    print(filename)

img = cv2.imread(image_test)
cv2.imshow("test image",img)
cv2.waitKey(0)
cv2.destroyAllWindows() 



sift = cv2.xfeatures2d.SIFT_create()

histogram_list = []

dico = []

def defineBOW_SIFT():
    #for leaf in train.id:
       # img = cv2.imread(img_path + str(leaf) + ".jpg")
    img = cv2.imread(image_test)
    cv2.imshow("test image",img)
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

    #plt.imshow(img)
    #plt.show()
        #kp, des = sift.detectAndCompute(img, None)
    kp, des = sift.detectAndCompute(img, None)

    #for d in des:
    dico.append(des)

    k = fixed_n_classes * 10
    #k = np.size(species) * 10

    #batch_size = np.size(os.listdir(img_path)) * 3
    #kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, verbose=1).fit(dico)
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size = 1 , verbose=1).fit(dico)

    # histogram creation

    kmeans.verbose = False


    #for leaf in train.id:
    img = cv2.imread(image_test)
    kp, des = sift.detectAndCompute(img, None)

    histo = np.zeros(k)
    nkp = np.size(kp)

    for d in des:
        idx = kmeans.predict([d])
        histo[idx] += 1/nkp # Because we need normalized histograms, I prefere to add 1/nkp directly

    histogram_list.append(histo)


#defineBOW_SIFT()
#print(histogram_list)

#img = cv2.imread(image_test)
#cv2.imshow("test",img)
#cv2.waitKey(0) 
#cv2.destroyAllWindows() 
