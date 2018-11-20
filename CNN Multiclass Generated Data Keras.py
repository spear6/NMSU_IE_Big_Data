# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 13:57:22 2018

@author: donov
"""

import numpy as np
import pandas as pd
import math
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy import stats
from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPool1D, Input, GlobalAveragePooling1D
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.utils import np_utils
#from sklearn.metrics import confusion_matrix
import keras.backend as K
from keras.utils import to_categorical
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

a=10000 #Normal Data Iterations
b=100 #Cyclic Data Iterations
c=100 #Systematic Data Iterations
d=100 #Upshift Iterations
e=100 #Downshift Iterations
f=100 #Uptrend Iterations
g=100 #Downtrend Iterations
total = a+b+c+d+e+f+g
mu=0
sigma=1
w = 50  #windown length 
bt = .405 # Cyclic trend size
ct = 1.53 # Systematic trend size
dt = .38 # Upshift trend size
et = .63 # Downshift trend size
ft = .93 # Uptrend trend size
gt = .58 # Downtrend trend size


#def weighted_mse(y_pred, y_true):
#    majority_weight= a/(a+b+c+d+e+f+g)
#    minority_weight_cyclic=b/(a+b+c+d+e+f+g)
#    minority_weight_systematic=c/(a+b+c+d+e+f+g)
#    minority_weight_Upshift=d/(a+b+c+d+e+f+g)
#    minority_weight_Downshift=e/(a+b+c+d+e+f+g)
#    minority_weight_Uptrend=f/(a+b+c+d+e+f+g)
#    minority_weight_Downtrend=g/(a+b+c+d+e+f+g)
        # Apply the weights
#    loss = K.mean(K.square((y_pred - y_true)*(y_true * majority_weight) + (1. - y_true) * minority_weight_cyclic) + (2. - y_true) * minority_weight_systematic),axis=-1)

        # Return the mean error
#    return loss

class_weights = {0 : 1/a, 1: 1/b, 2: 1/c, 3: 1/d, 4: 1/e, 5: 1/f, 6: 1/g}
                                             
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
Data2 = 0*[w,b] #Create null array for abnormal Cyclic data
Data3 = 0*[w,c] #Create null array for abnormal Systematic data
Data4 = 0*[w,d] #Create null array for abnormal Upshift data
Data5 = 0*[w,e] #Create null array for abnormal Downshift data
Data6 = 0*[w,f] #Create null array for abnormal Uptrend data
Data7 = 0*[w,g] #Create null array for abnormal Downtrend data
Data = 0*[w,a+b+c+d+e+f+g] #Create null array for concat data
x = 0*w
x_cyclic = 0*w 
x_systematic = 0*w 
x_upshift = 0*w 
x_downshift = 0*w
x_uptrend = 0*w 
x_downtrend = 0*w  


# Generates Normal data points # this is an extra step, but I want to replicate getting a random streams of w length
for i in range(0,a): #obviously filling a array of size(a,w) would be more efficient, but I wanted the normal/abnormal methods to match
    x = np.array(np.random.normal(mu, sigma, w))
    Data1 = np.concatenate((Data1, x), axis = 0) #concatenates the data and gives each row an index of 0(normal data)
    
Data1=np.reshape(Data1,(a,w))   
Data1 = pd.DataFrame(Data1) #creates a data frame of normal data   
Data1=Data1.assign(label=(np.zeros(a)))  # attaches 1 label to all normal row vectors    
Data1.label=Data1.label.astype(int) #converts the label to int = 1
    
s = np.arange(w)
    
# Generates cyclic Abnormal data points
for i in range(0,b):
    x_cyclic = np.random.normal(mu, sigma, w) + np.sin(np.pi*s/4)*bt
    Data2 = np.concatenate((Data2,x_cyclic), axis=0)
Data2 = np.reshape(Data2,(b,w))  # Data is created
Data2 = pd.DataFrame(Data2) #creates a data frame of abnormal data
Data2 = Data2.assign(label=(np.ones(b)))# attaches 1 label to all abnormal row vectors
Data2.label=Data2.label.astype(int) #converts the label to an integer 1 for cyclic abnormal vectors

#Generates systematic abnormal data points
for i in range(0,c):
    x_systematic = np.random.normal(mu,sigma,w) + ct*((-1)^s)
    Data3 = np.concatenate((Data3,x_systematic), axis=0)
Data3 = np.reshape(Data3,(c,w))  # Data is created
Data3 = pd.DataFrame(Data3) #creates a data frame of abnormal data
Data3 = Data3.assign(label=(2*np.ones(c)))# attaches 2 label to all abnormal row vectors
Data3.label=Data3.label.astype(int) #converts the label to an integer 1 for cyclic abnormal vectors

#Generates Upshift abnormal data points
for i in range(0,d):
    x_upshift = np.random.normal(mu,sigma,w) + dt*(np.ones(w))
    Data4 = np.concatenate((Data4,x_upshift), axis=0)
Data4 = np.reshape(Data4,(d,w))  # Data is created
Data4 = pd.DataFrame(Data4) #creates a data frame of abnormal data
Data4 = Data4.assign(label=(3*np.ones(d)))# attaches 2 label to all abnormal row vectors
Data4.label=Data4.label.astype(int) #converts the label to an integer 1 for cyclic abnormal vectors

#Generates Downshift abnormal data points
for i in range(0,e):
    x_downshift = np.random.normal(mu,sigma,w) - et*(np.ones(w))
    Data5 = np.concatenate((Data5,x_downshift), axis=0)
Data5 = np.reshape(Data5,(e,w))  # Data is created
Data5 = pd.DataFrame(Data5) #creates a data frame of abnormal data
Data5 = Data5.assign(label=(4*np.ones(e)))# attaches 2 label to all abnormal row vectors
Data5.label=Data5.label.astype(int) #converts the label to an integer 1 for cyclic abnormal vectors

#Generates Uptrend abnormal data points
for i in range(0,f):
    x_uptrend = np.random.normal(mu,sigma,w) + ft*s
    Data6 = np.concatenate((Data6,x_uptrend), axis=0)
Data6 = np.reshape(Data6,(f,w))  # Data is created
Data6 = pd.DataFrame(Data6) #creates a data frame of abnormal data
Data6 = Data6.assign(label=(5*np.ones(f)))# attaches 2 label to all abnormal row vectors
Data6.label=Data6.label.astype(int) #converts the label to an integer 1 for cyclic abnormal vectors

#Generates Uptrend abnormal data points
for i in range(0,g):
    x_downtrend = np.random.normal(mu,sigma,w) - gt*s
    Data7 = np.concatenate((Data7,x_downtrend), axis=0)
Data7 = np.reshape(Data7,(g,w))  # Data is created
Data7 = pd.DataFrame(Data7) #creates a data frame of abnormal data
Data7 = Data7.assign(label=(6*np.ones(g)))# attaches 2 label to all abnormal row vectors
Data7.label=Data7.label.astype(int) #converts the label to an integer 1 for cyclic abnormal vectors

#Concatenate the Labels    
Response_raw = np.concatenate((Data1.label,Data2.label,Data3.label,Data4.label,Data5.label,Data6.label,Data7.label), axis=0)
    
#Concatendate the Predictors
Data_raw = np.concatenate((Data1,Data2,Data3,Data4,Data5,Data6,Data7),axis=0) # concats the response dataframes
    #Datarr = Data[1:w]
    

    #print(Data.shape) #checks the shape of the data array 
Data_raw = pd.DataFrame(Data_raw)
Response_raw = pd.DataFrame(Response_raw)

#Data_raw.to_csv("DataGeneration.csv") 
#test_ext = int((.25*a) - (.25*b))        
#Data_ext = []        
#for i in range(0,test_ext):
#    if abtype == 1:
#                y_new = np.random.normal(mu,sigma,w) + t*s        # Up trend(+)
#                Data_ext = np.concatenate((Data_ext,y_new), axis = 0)    
#    if abtype == 3:
#                y_new = np.random.normal(mu,sigma,w) + t*(np.ones(w))  # Up shift(+)
#                Data_ext = np.concatenate((Data_ext,y_new), axis = 0)                    
#    if abtype == 5:
#                y_new =  np.random.normal(mu, sigma, w) + np.sin(np.pi*s/4)*t #Cyclical
#                Data_ext = np.concatenate((Data_ext,y_new), axis = 0)
#    if abtype == 6:
#                y_new = t * np.random.normal(mu,sigma,w)   # Stratification
#                Data_ext = np.concatenate((Data_ext,y_new), axis = 0)            
#Data_ext = np.reshape(Data_ext,(test_ext,w))  # Data is create
#Data_ext = pd.DataFrame(Data_ext) #creates a data frame of abnormal data
#Data_ext = Data_ext.assign(label=(np.zeros(test_ext)))# attaches 0 label to all abnormal row vectors
#Data_ext.label=Data_ext.label.astype(int) #converts the label to an integer 0 for abnormal vectors      
#y_ext = Data_ext.label
#y_ext = pd.DataFrame(y_ext)
#x_ext = Data_ext.iloc[:,:-1]


#Remove the labels and zscore the predictors
Data_raw = Data_raw.iloc[:,:-1]
Data_raw = stats.zscore(Data_raw)

#split the data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(Data_raw, Response_raw, test_size=.25) 

Y_test = to_categorical(Y_test)
Y_train = to_categorical(Y_train)       
        
#X_test = np.vstack([X_test, x_ext])
#Y_test = np.vstack([Y_test, y_ext])

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


#Define model architecture
 
#Use a sequential model
model = Sequential()
#1st Hidden Layer
model.add(Conv1D(filters = 20, kernel_size = 4,  padding = 'same', activation='relu', input_shape=(w,1)))
model.add(Conv1D(filters = 20, kernel_size = 4, padding = 'same',activation='relu'))
model.add(MaxPool1D(3))
model.add(Dropout(0.25))
# Second Hidden Layer
model.add(Conv1D(filters = 20, kernel_size = 3, padding = 'same', activation='relu'))
model.add(MaxPool1D(2))
model.add(Conv1D(filters = 20, kernel_size = 2, padding = 'same', activation='relu'))
model.add(Dropout(0.25))
model.add(GlobalAveragePooling1D())
#model.add(Dropout(0.25))
# Output
#model.add(Dense(7, activation='sigmoid'))
model.add(Dense(7, activation='softmax'))


model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])


reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.2, patience=2, min_lr=0.0001)
#tensorboard = TensorBoard(log_dir='./cnn_runs_tensorboard', histogram_freq=0, batch_size=16, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.fit(X_train, Y_train, batch_size=16, epochs=5, verbose = 1, class_weight = class_weights, callbacks=[reduce_lr])
scores = model.evaluate(X_test, Y_test, batch_size=16)

#Print verbose output
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.summary()


#pull prediction and test labels
y_predict = model.predict_classes(X_test)

y_truth = np.argmax(Y_test, axis=1)

#compute confusion matrix, sensitivity, specifity, and G-mean (for binary)
A = confusion_matrix(y_truth, y_predict)
print(A)

print('f1:',f1_score(y_truth, y_predict, average="macro"))
print('precision:', precision_score(y_truth, y_predict, average="macro"))
print('recall:',recall_score(y_truth, y_predict, average="macro"))
        
K.clear_session()
