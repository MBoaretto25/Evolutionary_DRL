# -*- coding: utf-8 -*-
"""
Function to Generate a NN Based on the Given Parameters 
@author: marco

Parameters to Tune:
    
X = [2,[8,16],'relu',[0.3]]
    
num_layers = [2,3]

num_units = [
            if layers == 2:
             0..[8,16],
             1..[32,16],
             2..[8,32],
             3..[16,32],
            if layers == 3:
             0..[8,16,32],
             1..[16,32,8],
             2..[8,32,8],
             3..[16,8,16]]

func_activ = ['relu','elu','selu',leaky_relu]

is_dropout =[                        
            if layers == 2:
             0..[1],
             1..[0.3],
             2..[0.5],
             3..[0.2],
            if layers == 3:
             0..[1, 1],
             1..[1,0.3],
             2..[.3,1],
             3..[.3,.5],
"""
import tensorflow as tf

def Create_Network(X,networkstate,networkaction,networkreward):

    num_layers = X[0];
    num_units  = X[1];
    func_activ = X[2];
    is_dropout = X[3];

    action_onehot = tf.one_hot(networkaction, 2, name="actiononehot");    
    
    if num_layers == 2:
        # The variable in our network: Weights and biases 
        w1 = tf.Variable(tf.random_normal([4            , num_units[0]], stddev=0.35), name="W1")
        w2 = tf.Variable(tf.random_normal([num_units[0] , num_units[1]], stddev=0.35), name="W2")
        w3 = tf.Variable(tf.random_normal([num_units[1] , 2]           , stddev=0.35), name="W3")
        b1 = tf.Variable(tf.zeros([num_units[0]]), name="B1")
        b2 = tf.Variable(tf.zeros([num_units[1]]), name="B2")
        b3 = tf.Variable(tf.zeros(2), name="B3")
        
        if func_activ == 'elu':
            layer1 = tf.nn.elu(tf.add(tf.matmul(networkstate,w1), b1), name="Result1")
            drop1  = tf.nn.dropout(layer1,is_dropout[0])
            layer2 = tf.nn.elu(tf.add(tf.matmul(drop1,w2), b2), name="Result2")
            predictedreward = tf.add(tf.matmul(layer2,w3), b3, name="predictedReward")
        elif func_activ == 'relu':
            layer1 = tf.nn.relu(tf.add(tf.matmul(networkstate,w1), b1), name="Result1")
            drop1  = tf.nn.dropout(layer1,is_dropout[0])
            layer2 = tf.nn.relu(tf.add(tf.matmul(drop1,w2), b2), name="Result2")
        elif func_activ == 'selu':
            layer1 = tf.nn.selu(tf.add(tf.matmul(networkstate,w1), b1), name="Result1")
            drop1  = tf.nn.dropout(layer1,is_dropout[0])
            layer2 = tf.nn.selu(tf.add(tf.matmul(drop1,w2), b2), name="Result2")
        elif func_activ == 'leaky_relu':
            layer1 = tf.nn.leaky_relu(tf.add(tf.matmul(networkstate,w1), b1), name="Result1")
            drop1  = tf.nn.dropout(layer1,is_dropout[0])
            layer2 = tf.nn.leaky_relu(tf.add(tf.matmul(drop1,w2), b2), name="Result2")
            
        predictedreward = tf.add(tf.matmul(layer2,w3), b3, name="predictedReward")
        
    else:    
        # The variable in our network: Weights and biases 
        w1 = tf.Variable(tf.random_normal([4            , num_units[0]], stddev=0.35), name="W1")
        w2 = tf.Variable(tf.random_normal([num_units[0] , num_units[1]], stddev=0.35), name="W2")
        w3 = tf.Variable(tf.random_normal([num_units[1] , num_units[2]], stddev=0.35), name="W3")
        w4 = tf.Variable(tf.random_normal([num_units[2] , 2]           , stddev=0.35), name="W4")
        b1 = tf.Variable(tf.zeros([num_units[0]]), name="B1")
        b2 = tf.Variable(tf.zeros([num_units[1]]), name="B2")
        b3 = tf.Variable(tf.zeros([num_units[2]]), name="B3")
        b4 = tf.Variable(tf.zeros(2), name="B4")
        
        if func_activ == 'elu':
            layer1 = tf.nn.elu(tf.add(tf.matmul(networkstate,w1), b1), name="Result1")
            drop1  = tf.nn.dropout(layer1,is_dropout[0])
            layer2 = tf.nn.elu(tf.add(tf.matmul(drop1,w2), b2), name="Result2")
            drop2  = tf.nn.dropout(layer2,is_dropout[1])
            layer3 = tf.nn.elu(tf.add(tf.matmul(drop2,w3), b3), name="Result3")
        elif func_activ == 'relu':
            layer1 = tf.nn.relu(tf.add(tf.matmul(networkstate,w1), b1), name="Result1")
            drop1  = tf.nn.dropout(layer1,is_dropout[0])
            layer2 = tf.nn.relu(tf.add(tf.matmul(drop1,w2), b2), name="Result2")
            drop2  = tf.nn.dropout(layer2,is_dropout[1])
            layer3 = tf.nn.relu(tf.add(tf.matmul(drop2,w3), b3), name="Result3")
        elif func_activ == 'selu':
            layer1 = tf.nn.selu(tf.add(tf.matmul(networkstate,w1), b1), name="Result1")
            drop1  = tf.nn.dropout(layer1,is_dropout[0])
            layer2 = tf.nn.selu(tf.add(tf.matmul(drop1,w2), b2), name="Result2")
            drop2  = tf.nn.dropout(layer2,is_dropout[1])
            layer3 = tf.nn.selu(tf.add(tf.matmul(drop2,w3), b3), name="Result3")
        elif func_activ == 'leaky_relu':
            layer1 = tf.nn.leaky_relu(tf.add(tf.matmul(networkstate,w1), b1), name="Result1")
            drop1  = tf.nn.dropout(layer1,is_dropout[0])
            layer2 = tf.nn.leaky_relu(tf.add(tf.matmul(drop1,w2), b2), name="Result2")
            drop2  = tf.nn.dropout(layer2,is_dropout[1])
            layer3 = tf.nn.leaky_relu(tf.add(tf.matmul(drop2,w3), b3), name="Result3")
            
        predictedreward = tf.add(tf.matmul(layer3,w4), b4, name="predictedReward")
        
    ## Learning 
    # Getting the Qreward which is a vector resultant from the multiplication of the predicted
    # rewards and the actions from the minibatch. Resultant(32,2)
    qreward = tf.reduce_sum(tf.multiply(predictedreward, action_onehot), reduction_indices = 1)
    # Calculate the loss function from the network reward(ground true) and the predicted qreward
    loss = tf.reduce_mean(tf.square(networkreward - qreward))
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.RMSPropOptimizer(0.0001).minimize(loss)
    merged_summary = tf.summary.merge_all()
    
    yield [optimizer,merged_summary,predictedreward]
