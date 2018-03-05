# -*- coding: utf-8 -*-
"""
From: http://www.pinchofintelligence.com/introduction-openai-gym-part-2-building-deep-q-network/
@author: marco
"""
## Imports
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import random
import Create_Network

start_time = time.time()

## Create gym environmnet 
env = gym.make('CartPole-v0')
observation = env.reset()

# Network input : Definition of the placeholders
networkstate = tf.placeholder(tf.float32, [None, 4], name="input")
networkaction = tf.placeholder(tf.int32, [None], name="actioninput")
networkreward = tf.placeholder(tf.float32,[None], name="groundtruth_reward")

X = [2,[8,16],'relu',[0.3]];
##
[optimizer,merged_summary,predictedreward] = Create_Network(X,networkstate,networkaction,networkreward)
##

## Var Initialization
sess = tf.InteractiveSession()
summary_writer = tf.summary.FileWriter('trainsummary',sess.graph)
sess.run(tf.global_variables_initializer())

## Replay 
replay_memory = [] # (state, action, reward, terminalstate, state_t+1)

initial_epsilon = 1.0
epsilon = initial_epsilon;
final_epsilon = 0.01
BATCH_SIZE = 32
GAMMA = 0.9
MAX_LEN_REPLAY_MEMORY = 10000
FRAMES_TO_PLAY = 10001
MIN_FRAMES_FOR_LEARNING = 100
summary = None
avg_loss_iter = []

for i_epoch in range(FRAMES_TO_PLAY):
    

    ## Without epsilon annealing
#    action = env.action_space.sample() 
    
    ### Epsilon annealing
    if  np.random.rand() <= epsilon: #generate a random value
        action = random.randrange(env.action_space.n)
    else: # or let the agent decide which one is best giving the actual state
        pred_reward = sess.run(predictedreward, feed_dict={networkstate:observation.reshape([1,4])})
        action = np.argmax(pred_reward[0])
        
    # Start Decaying the Epsilon in order to perform epsilon annealing
    # Decay the epsilon every 10t frame and until it reaches the minimum boundary
    if (epsilon > final_epsilon):
        epsilon -= (initial_epsilon - final_epsilon) / FRAMES_TO_PLAY
    
    #Get the newobservation and reward giving the action from the previous operation
    newobservation, reward, terminal, info = env.step(action)

    ### I prefer that my agent gets 0 reward if it dies
    if terminal: 
        reward = 0 
        
    ### Add the observation to our replay memory
    replay_memory.append((observation, action, reward, terminal, newobservation))
    
    ### Reset the environment if the agent died
    if terminal: 
        newobservation = env.reset()
    observation = newobservation
    
    ### Learn once we have enough frames to start learning
    if len(replay_memory) > MIN_FRAMES_FOR_LEARNING: 
        # From the memory variable, extract the training batch
        experiences = random.sample(replay_memory, BATCH_SIZE)
        totrain = [] # (state, action, delayed_reward)
        
        ### Calculate the predicted reward gibing the previous states from the batch
        nextstates = [var[4] for var in experiences]
        pred_reward = sess.run(predictedreward, feed_dict={networkstate:nextstates})    
            
        ### Set the "ground truth": the value our network has to predict:
        ### Calculate the Q(s,a) value giving the values from the batch.
        for index in range(BATCH_SIZE):
            state, action, reward, terminalstate, newstate = experiences[index]
            predicted_reward = max(pred_reward[index])
            
            if terminalstate:
                delayedreward = reward
            else:
                delayedreward = reward + GAMMA*predicted_reward
            totrain.append((state, action, delayedreward))
            
        ### Feed the train batch to the algorithm 
        states = [var[0] for var in totrain]
        actions = [var[1] for var in totrain]
        rewards = [var[2] for var in totrain]
        ## Calculate the loss function 
        _, l, summary = sess.run([optimizer, loss, merged_summary], feed_dict={networkstate:states, networkaction: actions, networkreward: rewards})

        avg_loss_iter.append(l)

        ### If our memory is too big: remove the first element
        if len(replay_memory) > MAX_LEN_REPLAY_MEMORY:
                replay_memory = replay_memory[1:]

        ### Show the progress 
        if i_epoch%100==1:
            summary_writer.add_summary(summary, i_epoch)
        if i_epoch%10==1:
            print("Epoch %d, loss: %f, e: %f" % (i_epoch,l,epsilon))
    
total_time = (time.time() - start_time)
print('--- Average Loss of Iterations = {:f}'.format(np.mean(avg_loss_iter)))
print('Total time = {:f} s / {:d}/m '.format(total_time,total_time/60))

