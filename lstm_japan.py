# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 14:18:33 2018

@author: donov
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from numpy import concatenate
import csv

#japan = pd.read_csv('japan_fuel_lstm.csv')

#print(japan)

#window = 10

#japan1 = japan.values
#japan1 = japan1.astype('float32')

def create_japan1(japan1, look_back=1):
	dataX, dataY = [], []
	for i in range(len(japan1)-look_back-1):
		a = japan1[i:(i+look_back)]
		dataX.append(a)
		dataY.append(japan1[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


#Read the japan dataset

japan = pd.read_csv('japan_fuel3.csv')
japan1 = japan.values
japan1 = japan1.astype('float32')

# normalize the japan1
scaler = MinMaxScaler(feature_range=(0, 1))
japan1 = scaler.fit_transform(japan1)

# split into train and test sets
train_size = int(len(japan1) * 0.67)
test_size = len(japan1) - train_size
train, test = japan1[0:train_size,:], japan1[train_size:len(japan1),:]

# reshape into X=t and Y=t+1
look_back = 20
trainX, trainY = create_japan1(train, look_back)
testX, testY = create_japan1(test, look_back)
# reshape input to be [samples, time steps, features]
#trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(16, input_shape=(look_back,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions


trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = np.empty_like(japan1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(japan1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(japan1)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(japan1))
plt.plot(trainPredictPlot, label = 'Train Predict')
plt.plot(testPredictPlot, label = 'Test Predict')
plt.title("Japan (2011) LSTM Prediction Plot")
plt.xlabel('Time Index')
plt.ylabel('Demand')
plt.legend(loc='upper right')
plt.savefig('Japan_LSTM_Toy.png')
plt.show()



