# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:33:58 2019

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

def tanh_mod(S,L,X_train,rho):
    X_train = np.add(L,(np.tanh(rho*np.size(X_train, axis=0)) * S))
    return X_train

def mahalanobis_dis(S,tao):
    mah_dist=[]
    for i in range(0,len(S),1):
        mean_vec = np.mean(S)
        cov_S = np.cov(np.transpose(S))
        mah = distance.mahalanobis(mean_vec,S[i,],cov_S)
        mah_dist= np.append(mah,mah_dist)
    mah_array = np.array([mah_dist,]*8).transpose()
    mah_array[mah_array>tao]=0
    S = np.multiply(mah_array,S) 
    S_meanvec = np.mean(S)
    S_global = np.mean(S_meanvec)
    S[S==0]=S_global
    return S 
    
def var_model(S,tao):
    sd_array=[]
    for i in range(0,len(S),1):
        mean_vec = np.mean(S)
        sd_S = np.sqrt(np.diagonal(np.cov(np.transpose(S))))
        std_error_vec = np.divide((mean_vec - S[i,]),sd_S)
        sd_array=np.append(std_error_vec,sd_array)
    sd_array= sd_array.reshape(len(S),8)
    sd_array[sd_array>tao]=0
    S=np.multiply(sd_array,S)
    S_meanvec = np.mean(S)
    S_global = np.mean(S_meanvec)
    S[S==0]=S_global
    return S

def max_eigenvals(S,tao,a):
    for i in range(a,len(S, axis=0),1):
        lam = np.linalg.eigvals(np.matmul(np.transpose(S[(i-b):(i+1),]),(S[(i-b):(i+1),])))
        lam_minus = np.linalg.eigvals(np.matmul(np.transpose(S[(i-b):i, ]),(S[(i-b):i, ])))
    if np.max(lam) - np.max(lam_minus) > tao:
        S[-1] = np.zeros(np.size(S,axis=1))
    else:
        S[-1] = S[-1]
    return S[-1]

def one_class(S):
    clf = OneClassSVM(nu=0.1,kernel='rbf',gamma=0.1)
    clf.fit(S)
    predicted = clf.predict(S)
    predicted=np.reshape(predicted,(predicted.shape[0],1))
    S = predicted*S
    return S

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
japan = pd.read_csv('japan_fuel2_pca.csv')
japan1 = japan.values
japan1 = japan1[0:199,]
japan1 = japan1.astype('float32')

# normalize the japan1
scaler = MinMaxScaler(feature_range=(0, 1))
japan1 = scaler.fit_transform(japan1)
X = japan1
y = japan1[:,0]


tscv = TimeSeriesSplit(n_splits=198)
rho = 0.1 #weighting factor for sparse set
tao = 0.05

pred_vec=[]

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    rpca = R_pca(X_train) #genertes S and L arrays for X_train
    L, S = rpca.fit(max_iter=10000)
    S = np.abs(S)
   
    if np.size(S,axis=0)<20:
        tanh_mod(S,rho)
        
    else:
       mahalanobis_dis(S,tao)   
   
    X_train = np.add(L,S)    
    
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
    
true_vec = y[1:199,]

def rmse(pred_vec, true_vec):
    return np.sqrt(((pred_vec - true_vec) ** 2).mean())

def R2adj(pred_vec,true_vec):
    y_bar = np.repeat(np.average(true_vec),np.size(true_vec))
    SST = np.sum((true_vec-y_bar)**2)
    SSTr = np.sum((pred_vec-y_bar)**2)
    R2 = SSTr/SST
    p=np.size(X,axis=1)
    n= np.size(y_bar)
    return 1-(1-R2)*(n-1)/(n-p-1)

#def MAPE(pred_vec,true_vec):
 #   y_true, y_pred = _check_1d_array(true_vec, pred_vec)
#    true_vec, pred_vec = np.array(true_vec), np.array(pred_vec)
#    return np.mean(np.abs((true_vec - pred_vec) / true_vec)) * 100

#pred_vec = np.mean(true_vec, axis=0)
 


rmse_val = rmse(np.array(pred_vec), np.array(true_vec))
print("rms error is: " + str(rmse_val))

R2adjval = R2adj(pred_vec,true_vec)
print("The R2adj is: " + str(R2adjval))

#MAPEval = MAPE(pred_vec,true_vec)
#print("The MAPE is: " + str(MAPEval))


plt.plot(pred_vec)
plt.plot(true_vec)
plt.show()

#plt.plot(S[:,0])
#plt.show()


plt.plot(np.square(true_vec-pred_vec))
plt.show()

bins = np.arange(-100, 100, 0.05) # fixed bin size

plt.xlim([min(S[:,0])-0.05, max(S[:,0])+0.05])

plt.hist(S[: ,0], bins=bins, alpha=0.5)
plt.title('Katrina Outlier Distribution (fixed bin size)')
plt.xlabel('variable X (bin size = 0.0.5)')
plt.ylabel('count')

plt.show()

plt.plot(S[:,0])
plt.show
