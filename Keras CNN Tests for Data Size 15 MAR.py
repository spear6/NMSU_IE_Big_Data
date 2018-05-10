# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 09:25:15 2018

@author: donov
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 20:10:38 2018

@author: donov
"""

##TEST PROGRAM FOR TS SHIFT DETECTION IN IMBALANCED SETS USING CNN##

##CALLED LIBRARIES##
import numpy as np
np.random.seed(123)  # for reproducibility
import pandas as pd
import math
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPool1D, Input, GlobalAveragePooling1D
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy import stats
from matplotlib import pyplot as plt

##SET PARAMETERS##
# a: majority class size
a= int
# b: minority class size
b= int
# mu: mean of normal class data
mu = 0.
# sigma : standard deviation of normal class data
sigma = 1.
# abtype: type of abnormalizy in the data (1-6)
abtype = 5

#TUNING VARIABLES##
# w: window length of the time series - starting value is 10
w = 25
# t: trend shift size of minority class data - starting value is 0.005
t = .5

import timeit


##RETURNS##
# G-mean array
Gmean_array = []


##CALLED FUNCTIONS##
#Create weighted mse loss function
def weighted_mse(y_pred, y_true):

    majority_weight=0.95
    minority_weight=0.05
        # Apply the weights
    loss = K.mean(K.square(y_pred - y_true)*((y_true * majority_weight) + (1. - y_true) * minority_weight), axis=-1)

        # Return the mean error
    return loss



# Use Loops to Define Arrays of Gmean and Specificity between w=10 to 50, and t=0.005 to t=0.1 
sizes = [500, 1000, 5000, 10000, 25000, 50000, 75000]
for i in sizes:
        a= round(i*0.95)
        b= round(i*0.05)
        start_time = timeit.default_timer()
        Data=[]
        Data1=[]
        Data2=[]
        #    Data1 = 0*[w,a] #Create null array for normal data
        Data2 = 0*[w,b] #Create null array for abnormal data
        Data = 0*[w,b+a] #Create null array for concat data
        y = 0*w #Create null vector for y values
        
        for i in range(0,a): #obviously filling a array of size(a,w) would be more efficient, but I wanted the normal/abnormal methods to match
            x = np.array(np.random.normal(mu, sigma, w))
            Data1 = np.concatenate((Data1, x), axis = 0) #concatenates the data and gives each row an index of 0(normal data)
    
        Data1=np.reshape(Data1,(a,w))
    
        Data1 = pd.DataFrame(Data1) #creates a data frame of normal data
    
        Data1=Data1.assign(label=(np.ones(a)))  # attaches 1 label to all normal row vectors 
    
        Data1.label=Data1.label.astype(int) #converts the label to int = 1
    
        s = np.arange(w)
    
    # Generates Abnormal data points
        for i in range(0,b):
            if abtype == 1:
                y = np.random.normal(mu,sigma,w) + t*s        # Up trend(+)
                Data2 = np.concatenate((Data2,y), axis = 0)    
            if abtype == 2:
                y = np.random.normal(mu,sigma,w) - t*s        # Downtrend(-)
                Data2 = np.concatenate((Data2,y), axis = 0)
            if abtype == 3:
                y = np.random.normal(mu,sigma,w) + t*(np.ones(w))  # Up shift(+)
                Data2 = np.concatenate((Data2,y), axis = 0)
            if abtype == 4:
                y = np.random.normal(mu,sigma,w) - t*(np.ones(w))  # Down shift(-)
                Data2 = np.concatenate((Data2,y), axis = 0)                    
            if abtype == 5:
                y =  np.random.normal(mu, sigma, w) + np.sin(np.pi*s/4)*t #Cyclical
                Data2 = np.concatenate((Data2,y), axis = 0)
            if abtype == 6:
                y = t * np.random.normal(mu,sigma,w)   # Stratification
                Data2 = np.concatenate((Data2,y), axis = 0)

        Data2 = np.reshape(Data2,(b,w))  # Data is created
        Data2 = pd.DataFrame(Data2) #creates a data frame of abnormal data
        Data2 = Data2.assign(label=(np.zeros(b)))# attaches 0 label to all abnormal row vectors
        Data2.label=Data2.label.astype(int) #converts the label to an integer 0 for abnormal vectors
    
    
        Response_raw = np.concatenate((Data1.label,Data2.label), axis=0)
    
        #Concatendate all Data
        Data_raw = np.concatenate((Data1,Data2),axis=0) # concats the two dataframes
        #Datarr = Data[1:w]
    
        #print(Data.shape) #checks the shape of the data array 
        Data_raw = pd.DataFrame(Data_raw)
        Response_raw = pd.DataFrame(Response_raw)
        #Data_raw.to_csv("DataGeneration.csv") 
   
        #Preprocessing
        Data = Data_raw.iloc[:, :-1] #This is a pandas function -  I may want to change to numpy

        Data = stats.zscore(Data)

        X_train, X_test, Y_train, Y_test = train_test_split(Data, Response_raw, test_size=0.25)  
        
        X_train = np.expand_dims(X_train, axis=2)
        X_test = np.expand_dims(X_test, axis=2)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        #CNN Model
        model = Sequential()
        #1st Hidden Layer
        model.add(Conv1D(filters = 100, kernel_size = 4, padding= 'same', activation='relu', input_shape=(w,1)))
        model.add(MaxPool1D(4, padding = 'same'))
        model.add(Conv1D(filters = 80, kernel_size = 4, padding = 'same', activation='relu'))
        model.add(MaxPool1D(3, padding = 'same'))
        model.add(Dropout(0.25))
        # Second Hidden Layer
        model.add(Conv1D(filters = 60, kernel_size = 3, padding= 'same', activation='relu'))
        model.add(Conv1D(filters = 40, kernel_size = 3, padding = 'same', activation='relu'))
        model.add(MaxPool1D(2, padding = 'same'))
        model.add(Dropout(0.25))
        model.add(GlobalAveragePooling1D())
        # Third Hidden Layer
        #model.add(Conv1D(filters = 150, kernel_size = 2, activation='relu'))
        #model.add(Conv1D(filters = 120, kernel_size = 2, activation='relu'))
        #model.add(GlobalAveragePooling1D())
        #model.add(Dropout(0.25))
        # Output
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

        reduce_lr = ReduceLROnPlateau(monitor='acc', factor=.05, patience=2, min_lr=0.000001)
        #tensorboard = TensorBoard(log_dir='./cnn_runs_tensorboard', histogram_freq=0, batch_size=16, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        #tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        model.fit(X_train, Y_train, batch_size=None, epochs=5, verbose=0, class_weight=None, callbacks=[reduce_lr])
        scores = model.evaluate(X_test, Y_test, batch_size=16)
        
        #print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        #model.summary()        
        
        y_predict = model.predict_classes(X_test)
        y_truth = Y_test

        #compute confusion matrix, sensitivity, specifity, and G-mean (for binary)
        A = confusion_matrix(y_truth, y_predict)
        #print(A)
        Sensitivity = A[0][0] / (A[0][0] + A[1][0])
        Specificity = A[1][1] / (A[1][1] + A[0][1])
        Gmean = np.sqrt(Sensitivity*Specificity)
        
        print("Confusion Matrix for Data Size:", a+b, "is:", A)
        print("Gmean for Data Size:" , a+b, "is:", Gmean)
        print("Sensitivity Matrix for Data Size:", a+b, "is:", Sensitivity)
        print("Specificity for Data Size:" , a+b, "is:", Sensitivity)
        
        #Specificity_array = np.append(Specificity_array,Specificity)
        del Data
        del Data1
        del Data2
        del y_predict
        del y_truth
        del model
        del Data_raw
        del Response_raw
        del x
        del y
        del A
        del s
        del X_train
        del Y_train
        del X_test
        del Y_test
        del Gmean
        del Sensitivity
        del Specificity
        K.clear_session()
        elapsed = timeit.default_timer() - start_time
        print("Computer Time for size:", a+b, "is:", elapsed)
        
        