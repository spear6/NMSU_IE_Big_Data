# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 14:41:54 2018

@author: donov
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 14:15:43 2018

@author: donov
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM , GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import csv

#japan = pd.read_csv('japan_fuel_lstm.csv')

#print(japan)

#window = 10

#katrina1 = japan.values
#katrina1 = katrina1.astype('float32')

def create_katrina1(katrina1, look_back=1):
	dataX, dataY = [], []
	for i in range(len(katrina1)-look_back-1):
		a = katrina1[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(katrina1[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


#Read the katrina dataset

katrina = pd.read_csv('katrina_fuel3_pca.csv')
katrina1 = katrina.values
katrina1 = katrina1.astype('float32')

# normalize the katrina1
scaler = MinMaxScaler(feature_range=(0, 1))
katrina1 = scaler.fit_transform(katrina1)

# split into train and test sets
train_size = int(len(katrina1) * 0.67)
test_size = len(katrina1) - train_size
train, test = katrina1[0:train_size,:], katrina1[train_size:len(katrina1),:]

# reshape into X=t and Y=t+1
look_back = 20
trainX, trainY = create_katrina1(train, look_back)
testX, testY = create_katrina1(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(16, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
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
trainPredictPlot = np.empty_like(katrina1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(katrina1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(katrina1)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(katrina1))
plt.plot(trainPredictPlot, label='Train Predict')
plt.plot(testPredictPlot, label = 'Test Predict')
plt.title("Katrina LSTM Prediction Plot")
plt.xlabel('Time Index')
plt.ylabel('Demand')
plt.legend(loc='upper right')

#plt.ylabel('Window Length')
plt.savefig('Katrina_LSTM_Toy.png')
plt.show()



