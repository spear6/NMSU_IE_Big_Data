# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 09:49:11 2018

@author: donov
"""

##CNN detection with wafer dataset from UCR Database (Much thanks to them for providing it!!!!)

#Load packages

import numpy as np
np.random.seed(123)  # for reproducibility
import pandas as pd
import math
import tensorflow as tf
import timeit
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
from keras.utils import to_categorical


K.clear_session()
#Load data
np.random.seed(1)
#Create weighted mse loss function
def weighted_mse(y_pred, y_true):

    majority_weight=0.90
    minority_weight=0.10 #True minority ratio for the Wafer dataset is 10.8%
        # Apply the weights
    loss = K.mean(K.square(y_pred - y_true)*((y_true * majority_weight) + (1. - y_true) * minority_weight), axis=-1)

        # Return the mean error
    return loss


start_time = timeit.default_timer()

wafer_TRAIN = np.genfromtxt('wafer_TRAIN',delimiter=",", skip_header =1)

#Y_train = wafer_TRAIN[:,:1]
#X_train = np.delete(wafer_TRAIN, [0], axis=1)
Y_train = wafer_TRAIN[:,:1]
X_train = np.delete(wafer_TRAIN, [0], axis=1)

#print("The Y_train dims are:", str(Y_train.shape))
#print("The X_train dims are:", str(X_train.shape))


wafer_TEST= np.genfromtxt('wafer_TEST',delimiter=",", skip_header =1)

#Y_test = wafer_TEST[:,:1]
#X_test = np.delete(wafer_TEST, [0], axis=1)
Y_test = wafer_TEST[:,:1]
X_test= np.delete(wafer_TEST, [0], axis=1)

print("The Y_test dims are:", str(Y_test.shape))
print("The X_test dims are:", str(X_test.shape)) 

# Expand the dims for the NN CONV1D
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2) 
print("The expanded X_train dims are:", str(X_train.shape))
print("The expanded X_test dims are:", str(X_test.shape))

#Convert predictors to float 32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#Standardize predictors
X_test=stats.zscore(X_test)
X_train=stats.zscore(X_train)



    #CNN Model
model = Sequential()
   #1st Hidden Layer
model.add(Conv1D(filters = 100, kernel_size = 4, padding= 'same', activation='tanh', input_shape=(152,1)))
model.add(MaxPool1D(4, padding = 'same'))
model.add(Conv1D(filters = 90, kernel_size = 4, padding = 'same', activation='relu'))
model.add(MaxPool1D(3, padding = 'same'))
model.add(Dropout(0.25))
    # Second Hidden Layer
model.add(Conv1D(filters = 80, kernel_size = 3, padding= 'same', activation='tanh'))
model.add(Conv1D(filters = 70, kernel_size = 3, padding = 'same', activation='relu'))
model.add(MaxPool1D(1, padding = 'same'))

model.add(GlobalAveragePooling1D())
    # Third Hidden Layer
    #model.add(Conv1D(filters = 150, kernel_size = 2, activation='relu'))
    #model.add(Conv1D(filters = 120, kernel_size = 2, activation='relu'))
    #model.add(GlobalAveragePooling1D())
    #model.add(Dropout(0.25))
    # Output
class_weight = {0 : 1, 1: 10}

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='acc', factor=.05, patience=2, min_lr=0.000001)
#tensorboard = TensorBoard(log_dir='./cnn_runs_tensorboard', histogram_freq=0, batch_size=16, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.fit(X_train, Y_train, batch_size=None, epochs=20, verbose=1, class_weight=None, callbacks=[reduce_lr])
scores = model.evaluate(X_test, Y_test, batch_size=16)
        
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#model.summary()        

#print(history.history.keys())
# summarize history for accuracy
#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()
# summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()

        
y_predict = model.predict_classes(X_test)
y_truth = Y_test

    #compute confusion matrix, sensitivity, specifity, and G-mean (for binary)
A = confusion_matrix(y_truth, y_predict)
#print(A)
Sensitivity = A[0][0] / (A[0][0] + A[1][0])
Specificity = A[1][1] / (A[1][1] + A[0][1])
Gmean = np.sqrt(Sensitivity*Specificity)
        
print("The confusion matrix is:", A)
print("The Gmean is:", Gmean)
print("The Sensitivity is:", Sensitivity)
print("The Specificity is:". Specificity)
    #print("Confusion Matrix for Data Size:", str(X_test.shape), "is:", A)
    #print("Gmean for Data Size:" , str(X_test.shape), "is:", Gmean)
    #print("Sensitivity Matrix for Data Size:", str(X_test.shape), "is:", Sensitivity)
    #print("Specificity for Data Size:" , str(X_test.shape), "is:", Sensitivity)

    #print(A)
   
elapsed = timeit.default_timer() - start_time
print("Computer Time is:", elapsed)

#%%
yoga_TEST= np.genfromtxt('UCR_TS_Archive_2015\yoga\yoga_TEST.csv',delimiter=",", skip_header =1)

#Y_train = wafer_TRAIN[:,:1]
#X_train = np.delete(wafer_TRAIN, [0], axis=1)
Y_test = yoga_TEST[:,:1]
X_test = np.delete(yoga_TEST, [0], axis=1)

print("The Y_test dims are:", str(Y_test.shape))
print("The X_test dims are:", str(X_test.shape))


yoga_TRAIN=np.genfromtxt('UCR_TS_Archive_2015\yoga\yoga_TRAIN.csv',delimiter=",", skip_header =1)

#Y_test = wafer_TEST[:,:1]
#X_test = np.delete(wafer_TEST, [0], axis=1)
Y_train = yoga_TRAIN[:,:1]
X_train = np.delete(wafer_TRAIN, [0], axis=1)

print("The Y_train dims are:", str(Y_train.shape))
print("The X_train dims are:", str(X_train.shape)) 

# Expand the dims for the NN CONV1D
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2) 
print("The expanded X_train dims are:", str(X_train.shape))
print("The expanded X_test dims are:", str(X_test.shape))

#Convert predictors to float 32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#Standardize predictors
X_test=stats.zscore(X_test)
X_train=stats.zscore(X_train)



    #CNN Model
model = Sequential()
   #1st Hidden Layer
model.add(Conv1D(filters = 100, kernel_size = 4, padding= 'same', activation='tanh', input_shape=(426,1)))
model.add(MaxPool1D(4, padding = 'same'))
model.add(Conv1D(filters = 90, kernel_size = 4, padding = 'same', activation='relu'))
model.add(MaxPool1D(3, padding = 'same'))
model.add(Dropout(0.25))
    # Second Hidden Layer
model.add(Conv1D(filters = 80, kernel_size = 3, padding= 'same', activation='tanh'))
model.add(Conv1D(filters = 70, kernel_size = 3, padding = 'same', activation='relu'))
model.add(MaxPool1D(2, padding = 'same'))
model.add(Dropout(0.25))
model.add(GlobalAveragePooling1D())
    # Third Hidden Layer
    #model.add(Conv1D(filters = 150, kernel_size = 2, activation='relu'))
    #model.add(Conv1D(filters = 120, kernel_size = 2, activation='relu'))
    #model.add(GlobalAveragePooling1D())
    #model.add(Dropout(0.25))
    # Output
#class_weight = {0 : 1, 1: 10}

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor='acc', factor=.05, patience=2, min_lr=0.000001)
#tensorboard = TensorBoard(log_dir='./cnn_runs_tensorboard', histogram_freq=0, batch_size=16, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
#tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

scores = model.fit(X_train, Y_train, batch_size=None, epochs=100, verbose=1, class_weight=None, callbacks=[reduce_lr])
scores = model.evaluate(X_test, Y_test, batch_size=16)
        
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#model.summary()        
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

        
y_predict = model.predict_classes(X_test)
y_truth = Y_test

    #compute confusion matrix, sensitivity, specifity, and G-mean (for binary)
A = confusion_matrix(y_truth, y_predict)
#print(A)
Sensitivity = A[0][0] / (A[0][0] + A[1][0])
Specificity = A[1][1] / (A[1][1] + A[0][1])
Gmean = np.sqrt(Sensitivity*Specificity)
        
print("The confusion matrix is:", A)
print("The Gmean is:", Gmean)
    #print("Confusion Matrix for Data Size:", str(X_test.shape), "is:", A)
    #print("Gmean for Data Size:" , str(X_test.shape), "is:", Gmean)
    #print("Sensitivity Matrix for Data Size:", str(X_test.shape), "is:", Sensitivity)
    #print("Specificity for Data Size:" , str(X_test.shape), "is:", Sensitivity)

#print(A)   
    
