# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:50:26 2018

@author: donov
"""

import numpy as np
np.random.seed(123)  # for reproducibility

import pandas as pd
import math
import tensorflow as tf


def GenerateData(w, abtype, t):
                                          
    a=9950
    b=50
    mu=0
    sigma=1                                             
# INPUT:                                                                                                                                     
# w: window length    
# abtype: Abnormal type (e.g. Uptrend, Downtrend, Uphift,Downshift, Cyclic, Systematic, Stratification)
# t: parameter of abnormal pattern   
# a: the size of Normal class
# b: the size of abnormal class                                      
# mu for random numbers
# sigma for random numbers
    
# OUTPUT:                                                                                          
# Data: Time series data with label and weights, If we run SVM, the weights for all data samples are one.  

    Data1 = 0*[w,a] #Create null array for normal data
    Data2 = 0*[w,b] #Create null array for abnormal data
    Data = 0*[w,b+a] #Create null array for concat data
    y = 0*w #Create null vector for y values

# Generates Normal data points # this is an extra step, but I want to replicate getting a random streams of w length
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
            y =  np.random.normal(mu, sigma, w) + np.sin(np.pi*s/4)*t #np.random.normal(mu, sigma, w) + np.abs(
            Data2 = np.concatenate((Data2,y), axis = 0)
        if abtype == 6:
            y = np.random.normal(mu,sigma,w) + t*((-1)^s)   # Systematic
            Data2 = np.concatenate((Data2,y), axis = 0)

    Data2 = np.reshape(Data2,(b,w))  # Data is created
    Data2 = pd.DataFrame(Data2) #creates a data frame of abnormal data
    Data2 = Data2.assign(label=(np.zeros(b)))# attaches 0 label to all abnormal row vectors
    Data2.label=Data2.label.astype(int) #converts the label to an integer 0 for abnormal vectors
    
    global Response_raw
    Response_raw = np.concatenate((Data1.label,Data2.label), axis=0)
    
    #Concatendate all Data

    global Data_raw
    Data_raw = np.concatenate((Data1,Data2),axis=0) # concats the two dataframes
    #Datarr = Data[1:w]
    

    #print(Data.shape) #checks the shape of the data array 
    Data_raw = pd.DataFrame(Data_raw)
    Response_raw = pd.DataFrame(Response_raw)
    #Data_raw.to_csv("DataGeneration.csv") 
   
    return Response_raw, Data_raw, w #need to understand how to use w in the input to conv1D
    
    
#%%  
#split the data into test and train
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy import stats
from matplotlib import pyplot as plt
from keras.utils import np_utils
    
Data = Data_raw.iloc[:, :-1] #This is a pandas function -  I may want to change to numpy

Data = stats.zscore(Data)

X_train, X_test, Y_train, Y_test = train_test_split(Data, Response_raw, test_size=1/3)

# Print data shapes for reference

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))


 # Format shapes for 1D Convolutional Network

# process the data to fit in a keras 1D CNN properly
# input data needs to be (N, C, X, Y) - shaped where
# N - number of samples
# C - number of channels per sample
# (X, Y) - sample size 

# Preprocess input data (changes (n,c) to (n,c,1))
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

print ("X_train shape expanded is: " + str(X_train.shape))
print ("Y_train shape is: " + str(Y_train.shape))
print ("X_test shape expanded is: " + str(X_test.shape))
print ("Y_test shape is: " + str(Y_test.shape))



import keras.backend as K

def weighted_mse(y_pred, y_true):

    majority_weight=0.95
    minority_weight=0.05
        # Apply the weights
    loss = K.mean(K.square((y_pred - y_true)*(y_true * majority_weight) + (1. - y_true) * 1), axis=-1)

        # Return the mean error
    return loss

class_weight = {0 : 0.995, 1: 0.005}
 #%%
#Define model architecture
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPool1D, Input, GlobalAveragePooling1D
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.utils import np_utils

 
#Use a sequential model
model = Sequential()
#1st Hidden Layer
model.add(Conv1D(filters = 180, kernel_size = 4,  padding = 'same', activation='relu', input_shape=(20,1)))
model.add(Conv1D(filters = 100, kernel_size = 4, padding = 'same',activation='relu'))
model.add(MaxPool1D(3))
model.add(Dropout(0.25))
# Second Hidden Layer
model.add(Conv1D(filters = 80, kernel_size = 3, activation='tanh'))
model.add(MaxPool1D(2))
model.add(Conv1D(filters = 60, kernel_size = 2, activation='tanh'))
model.add(GlobalAveragePooling1D())
#model.add(Dropout(0.25))
# Output
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])


reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.2, patience=2, min_lr=0.0001)
#tensorboard = TensorBoard(log_dir='./cnn_runs_tensorboard', histogram_freq=0, batch_size=16, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.fit(X_train, Y_train, batch_size=16, epochs=5, class_weight = None, callbacks=[reduce_lr])
scores = model.evaluate(X_test, Y_test, batch_size=16)

#Print verbose output
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.summary()

#%%
from sklearn.metrics import confusion_matrix

#pull prediction and test labels
y_predict = model.predict_classes(X_test)
y_truth = Y_test

#compute confusion matrix, sensitivity, specifity, and G-mean (for binary)
A = confusion_matrix(y_truth, y_predict)
print(A)
Sensitivity = A[0][0] / (A[0][0] + A[1][0])
Specificity = A[1][1] / (A[1][1] + A[0][1])
Minority_Accuracy = A[0][0] / (A[0][0]+A[0][1])

Gmean = np.sqrt(Sensitivity*Specificity)
print("Precision is: ",Minority_Accuracy)
print("Sensitivity is: ", Sensitivity)
print("Specificity is: ", Specificity)
#print("Sensitivity is: ",Sensitivity)
print("The Gmean is:", Gmean)
        
K.clear_session()
