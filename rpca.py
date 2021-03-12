# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:19:35 2019

@author: donov
"""


from __future__ import division, print_function

import numpy as np

try:
    from pylab import plt
except ImportError:
    print('Unable to import pylab. R_pca.plot_fit() will not work.')

try:
    # Python 2: 'xrange' is the iterative version
    range = xrange
except NameError:
    # Python 3: 'range' is iterative - no need for 'xrange'
    pass


class R_pca:

    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * self.norm_p(self.D, 2))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def norm_p(M, p):
        return np.sum(np.power(M, p))

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.norm_p(np.abs(self.D), 2)

        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.norm_p(np.abs(self.D - Lk - Sk), 2)
            iter += 1
            if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
                print('iteration: {0}, error: {1}'.format(iter, err))

        self.L = Lk
        self.S = Sk
        return Lk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True):

        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        print('ymin: {0}, ymax: {1}'.format(ymin, ymax))

        numplots = np.min([n, nrows * ncols])
        plt.figure()

        for n in range(numplots):
            plt.subplot(nrows, ncols, n + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[n, :] + self.S[n, :], 'r')
            plt.plot(self.L[n, :], 'b')
            if not axis_on:
                plt.axis('off')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras as K
import keras.backend as K

# generate low rank synthetic data
japan = pd.read_csv('japan_fuel_lstm.csv')
japan1 = japan.values
japan1 = japan1.astype('float32')

# normalize the japan1
scaler = MinMaxScaler(feature_range=(0, 1))
japan1 = scaler.fit_transform(japan1)
X = japan1
y = japan1[:,0]


tscv = TimeSeriesSplit(n_splits=1382)
rho = .1 #weighting factor for sparse set

pred_vec=[]

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    rpca = R_pca(X_train) #genertes S and L arrays for X_train
    L, S = rpca.fit(max_iter=10000)
#    X_train = L + rho * S
    X_train = L + np.tanh(rho_iter*np.size(X_train, axis=0)) * S
    
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
# fit network
    history = model.fit(X_train, y_train, epochs=200, batch_size=50, validation_data=(X_test, y_test), verbose=0, shuffle=False)

    testPredict = model.predict(X_test)
    
    pred_vec = np.append(pred_vec,testPredict)
    
    K.clear_session()
# plot history
    
true_vec = y[1:1384,]

def rmse(pred_vec, true_vec):
    return np.sqrt(((pred_vec - true_vec) ** 2).mean())

rmse_val = rmse(np.array(pred_vec), np.array(true_vec))
print("rms error is: " + str(rmse_val))




# visually inspect results (requires matplotlib)
#rpca.plot_fit()
#plt.show()

plt.plot(pred_vec)
plt.plot(true_vec)
plt.show()

plt.plot(np.square(true_vec-pred_vec))
plt.show()

#plt.plot(S[:,0])
#plt.show()

#mean_vec = np.full((1384, 1), np.mean(true_vec))

#rmse_mean =rmse(np.array(mean_vec),np.array(true_vec))

#rmse_mean