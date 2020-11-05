import pandas as pd
import numpy as np
from numpy import where
# import matplotlib.pyplot as plt
# %matplotlib inline

import tensorflow as tf
from sklearn import preprocessing

# Linear Regression/MLP functions 

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from math import sqrt

from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model
import seaborn as sns

def MLP_function(X,X_train,X_test,y_train,y_test,num_neuron,bsize,num_epoch,val):
    # sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.transform(X_test)
    input_layer = Input(shape=(X.shape[1],))
    dense_layer_1 = Dense(num_neuron, activation='relu')(input_layer)
    output = Dense(1)(dense_layer_1)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss="mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
    history = model.fit(X_train, y_train, batch_size=bsize, epochs=num_epoch, verbose=1, validation_split=val)
    pred_MLP = model.predict(X_test)
    
    num_rows, k = X.shape
    n = len(y_test)
    select_pred_MLP= pred_MLP[:,0]
    SSE = sum((y_test-select_pred_MLP)**2)
    R2_MLP = r2_score(y_test, select_pred_MLP)
    AR2_MLP = 1-((1-R2_MLP)*(n-1)/(n-k-1))


   # print('SSE: %3f   R2: %3f   Adjusted R2: %3f' % (SSE,R2_MLP,AR2_MLP) )
    print(f'SSE: {SSE}')
    print(f'R2: {R2_MLP}')
    print(f'Adjusted R2: {AR2_MLP}')

