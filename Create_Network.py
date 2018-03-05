# -*- coding: utf-8 -*-
"""
Function to Generate a NN Based on the Given Parameters 
@author: marco

Parameters to Tune:
    
X = [2 , [8,16] , 'relu' , 'Adam' , 0 , 0.3]
    
num_layers = [2,3]

num_units = [
            if layers == 2:
             0..[16,16],
             1..[32,16],
             2..[8,32],
             3..[16,32],
            if layers == 3:
             0..[8,16,32],
             1..[16,32,8],
             2..[8,32,8],
             3..[16,8,16]]

activ_fun = ['relu','linear','selu','softmax']

optimizer = ['adam','SGD','adadelta','rmsprop']

w_droput  = (0,1)                      

*testes foram feitos com batch_normalization mas deixxava o alg mto lento

"""
from keras.models import Sequential
from keras.layers.core import Dropout
from keras.layers import Dense, BatchNormalization
from keras.optimizers import Adam,SGD,Adadelta,RMSprop

def Create_Network(X,state_size,action_size,learning_rate):

    ## Decompose Input Vector
    num_layers   = X[0]; # [2,3]
    num_units    = X[1]; # Por enquanto as unit serão iguais
    activ_fun    = X[2]; # ['relu','linear','selu','softmax']
    optimizer    = X[3]; # ['adam','SGD','adadelta','rmsprop']
    w_dropout    = X[4]; # (0,1)
    
    
    model = Sequential()
    
    if num_layers == 2:
        
        ## 1st Layer
        model.add(Dense(num_units, input_dim=state_size, activation=activ_fun, ))        
        model.add(Dropout(w_dropout))
        ## 2nd Layer
        model.add(Dense(num_units, activation=activ_fun))        
        model.add(Dropout(w_dropout))

    else:
        ## 1st Layer
        model.add(Dense(num_units, input_dim=state_size, activation=activ_fun))
        model.add(Dropout(w_dropout))
        
        ## 2nd Layer
        model.add(Dense(num_units, activation=activ_fun))        
        model.add(Dropout(w_dropout))
        
        ## 3rd Layer
        model.add(Dense(num_units, activation=activ_fun))        
        model.add(Dropout(w_dropout))   
        
    ## Output Layer
    model.add(Dense(action_size, activation='linear'))
    ## Compile
    model.compile(loss='mse', optimizer=optimizer)
    
    return model