# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:48:58 2018

@author: donov
"""

#Load all required backends and functions

import numpy as np
np.random.seed(123)  # for reproducibility
import pandas as pd
import math
import tensorflow as tf
import keras as K
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPool1D, Input, GlobalAveragePooling1D
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix, silhouette_score 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy import stats
from matplotlib import pyplot as plt

mu = 0
sigma = 1
abtype = 6
a=9500
b=500

w=25
t=0.06

def weighted_mse(y_pred, y_true):

    majority_weight=0.9500
    minority_weight=0.0500
        # Apply the weights
    loss = K.mean(K.square((y_pred - y_true)*(y_true * majority_weight) + (1. - y_true) * minority_weight), axis=-1)

        # Return the mean error
    return loss

# This is the function to manually call the data generation

                                                                                       
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
        y =  np.random.normal(mu, sigma, w) * (2 * np.sin(np.linspace(mu, np.pi, w))) #np.random.normal(mu, sigma, w) + np.abs(
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
        
#        X_test = np.vstack([X_test, x_ext])
#        Y_test = np.vstack([Y_test, y_ext])
        
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

    
#Use a sequential model
model = Sequential()
#1st Hidden Layer
model.add(Conv1D(filters = 80, kernel_size = 4, activation='relu', input_shape=(w,1)))
model.add(Conv1D(filters = 80, kernel_size = 4, activation='relu'))
model.add(MaxPool1D(3))
model.add(Dropout(0.25))
# Second Hidden Layer
model.add(Conv1D(filters = 80, kernel_size = 2, activation='tanh'))
model.add(Conv1D(filters = 80, kernel_size = 2, activation='tanh'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.25))
# Output
model.add(Dense(1, activation='sigmoid'))
# End CNN structure

# Model parameters and metrics
model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

#Define Call backs
reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.2, patience=2, min_lr=0.0001)
#tensorboard = TensorBoard(log_dir='./cnn_runs_tensorboard', histogram_freq=0, batch_size=16, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

#Run Model
model.fit(X_train, Y_train, batch_size=16, epochs=10, verbose = 0,class_weight = None, callbacks=[reduce_lr])
scores = model.evaluate(X_test, Y_test, batch_size=16)

#Print verbose output
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#model.summary()


## Call and Generate Data
#pull prediction and test labels
y_predict = model.predict_classes(X_test)
y_truth = Y_test



#compute confusion matrix, sensitivity, specifity, and G-mean (for binary)
A = confusion_matrix(y_truth, y_predict)
print(A)
A = np.array(A)
A=np.reshape(A,[2,2])
        #print(A)
Accuracy = (A[0,0] + A[1,1]) / (A[0,0] + A[0,1]+A[1,1] + A[1,0])
Sensitivity = A[0,0] / (A[0,0] + A[0,1])
Specificity = A[1,1] / (A[1,1] + A[1,0])
Gmean = np.sqrt(Sensitivity*Specificity)

print("Accuracy is: ",Accuracy)
print("Specificity is: ",Specificity)
print("Sensitivity is: ",Sensitivity)
print("The Gmean is:", Gmean)

K.clear_session()