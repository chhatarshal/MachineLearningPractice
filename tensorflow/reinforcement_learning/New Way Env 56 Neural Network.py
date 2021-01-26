#!/usr/bin/env python
# coding: utf-8

# In[113]:


import gym
from gym import wrappers

import numpy as np 
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

from tensorflow import keras


# In[114]:


class Env56:
    def __init__(self, final_state_val):
        self.action_space = 4 # up and down , left , right 0, 1, 2, 3
        self.state_space = 56 # 
        self.current_state = int(0)
        self.q_val = np.zeros([self.state_space, self.action_space])
        self.final_state = final_state_val
    def reset(self):
        self.current_state = 0
        return self.current_state
    
    def getNextState(self, action):
        if action == 0 and int(env.current_state) in [i for i in range(8)]:
            return self.current_state
        if action == 1 and int(env.current_state) in [i for i in range(48, 56)]:
            return self.current_state
        if action == 2 and int(env.current_state) == 0:
            return self.current_state
        if action == 3 and int(env.current_state) == 55:
            return self.current_state
        if action == 2:
            next_ = int(self.current_state) - 1
        if action == 3:
            next_ = int(self.current_state) + 1
        if action == 0:
            next_ = int(self.current_state) - 8
        if action == 1:
            next_ = int(self.current_state) + 8
        
        if next_ > 55 or next_ < 0:
            print(self.current_state)
            print(action)
            
        return next_
            
    def step(self, action):
        next_state = self.getNextState(action)
        self.current_state = str(next_state)
        done = (int(self.current_state) == int(self.final_state))
        reward = 0
        if done:
            reward = 1
        return (int(next_state),reward, done, "info--")


# In[115]:


env = Env56(27)


# In[116]:


checkpoint = ModelCheckpoint('model/model_Env56.h5', monitor='val_loss',verbose=1, save_best_only=True)
no_of_observations = 10000
min_score = 0


# In[117]:


# generate the training data 
def generate_training_data(no_of_episodes):
    print('generating training data')
    # initize the environment
   
    X = []
    y = []
    left = 0
    right = 0

    for i_episode in range(no_of_episodes):
        prev_observation = env.reset()
        score = 0
        X_memory  = []
        y_memory = []
        steps = 0
        for t in range(no_of_observations):
            action = random.randrange(0,4)
            
            ## debugging code
            
            new_observation,reward,done,info = env.step(action)
            score = score + reward
            steps = steps + 1
            X_memory.append(prev_observation)
            y_memory.append(action)
            prev_observation = new_observation
            if done:
                for data in X_memory:
                    X.append(data)
                for data in y_memory:
                    y.append(data)
                print('episode : ',i_episode, ' score : ',score, 'steps: ', steps)
                break
        env.reset()
    #debugging code
   
    # converting them into numpy array
    X = np.asarray(X)
    y =np.asarray(y) 
    #print(X)
   # print(y)
    # saving the numpy array
    np.save('data/X',X)
    np.save('data/y',y)
    
    # printing the size
    print('shape of X: ',X.shape)
    print('shape of target labels', y.shape)

# defines the model to be trained


# In[118]:


def get_model():
    model = Sequential()
    model.add(Dense(128, input_dim=1))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(.2))
    
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(.2))

    model.add(Dense(4))
    model.add(Activation('softmax'))
    
    model.summary()
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    return model


# In[119]:


# trains the model
def train_model(model):
    # loading the training data from the disk
    X= np.load('data/X.npy')
    y = np.load('data/y.npy')
    # making train test split 
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .2, random_state = 42)
    print('X_train: ',X_train.shape)
    print('y_train:', y_train.shape)
    print('X_test: ', X_test.shape)
    print('y_test: ', y_test.shape)
    # training the model
   # model.fit(X_train,y_train,validation_data = [X_test,y_test],verbose = 1,
   # callbacks=[checkpoint],
   # epochs= 20, batch_size = 10000,shuffle =True)
    # returns the model
    model.fit(X_train,y_train,  epochs= 1000, batch_size = 10000,shuffle =True)
    return model


# In[120]:


generate_training_data(100000)


# In[121]:


model = get_model()


# In[122]:


model.predict([0])


# In[123]:


model = train_model(model)


# In[ ]:





# In[ ]:




