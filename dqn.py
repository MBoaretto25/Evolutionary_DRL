# -*- coding: utf-8 -*-
import random
import numpy as np
from collections import deque
from Create_Network import Create_Network  
import gym

def dqn(x):
    
#    X_inp=[[0 , 120 , 1 , 1 , 0.3],[1 , 12 , 3 , 2 , 0.4]]
    # Initialize Vars
    v1 = [2,3]
    activ_fun = ['relu','linear','selu','softmax'];
    optimizer = ['adam','SGD','adadelta','rmsprop'];
    EPISODES = 50
    x_aux = np.zeros([4],dtype = int)
    
    
    x_aux[0] = x[0]
    x_aux[1] = x[1]
    x_aux[2] = x[2]
    x_aux[3] = x[3]    
   
    par_inp = [ v1[x_aux[0]],
                    x_aux[1],
         activ_fun[x_aux[2]],
         optimizer[x_aux[3]],
                    x[4]]
    
    class DQNAgent:
        def __init__(self, state_size, action_size):
            self.state_size = state_size
            self.action_size = action_size
            self.memory = deque(maxlen=2000)
            self.gamma = 0.90   # discount rate
            self.initial_epsilon = 1.0
            self.epsilon = 1.0  # exploration rate
            self.epsilon_min = 0.1
            self.epsilon_decay = 0.995
            self.learning_rate = 0.01
            self.model = self._build_model()
    
        def _build_model(self):
            # Neural Net for Deep-Q learning Model
            
            model = Create_Network(par_inp,self.state_size,self.action_size,self.learning_rate)
            return model
    
        def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))
    
        def act(self, state):
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])  # returns action
    
        def replay(self, batch_size):
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = (reward + self.gamma *
                              np.amax(self.model.predict(next_state)[0]))
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
            if self.epsilon > self.epsilon_min:
                self.epsilon -= (self.initial_epsilon - self.epsilon_min) / EPISODES
    
        def load(self, name):
            self.model.load_weights(name)
    
        def save(self, name):
            self.model.save_weights(name)

    env = gym.make('CartPole-v1')
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    done = False
    batch_size = 32
    scores_avg = []
    
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for i in range(200):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                scores_avg.append(i)
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)       
    
    del agent
    env.close()
    return np.mean(scores_avg) #Scores_Hist