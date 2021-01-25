#!/usr/bin/env python
# coding: utf-8

# In[310]:


import os
import numpy as np

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# In[311]:


class Env56:
    def __init__(self, final_state_val):
        self.action_space = 4 # up and down , left , right 0, 1, 2, 3
        self.state_space = 56 # 
        self.current_state = int(0)
        self.q_val = np.zeros([self.state_space, self.action_space])
        self.final_state = final_state_val
    def reset(self):
        self.current_state = 0
        return np.array([self.current_state])
    
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


# In[312]:


env = Env56(27)


# In[313]:


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_classes, input_dims):
        super(DeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
      
        layer1 = F.relu(self.fc1(data))
        layer2 = F.relu(self.fc2(layer1))
        layer3 = self.fc3(layer2)

        return layer3


# In[314]:


import numpy as np

class ReplayBuffer(object):
    #ReplayBuffer(mem_size, input_dims, n_actions)
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, input_shape),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        #print('store transition ')
        #print(state)
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


# In[315]:


class DQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, 1, n_actions)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    #name=self.env_name+'_'+self.algo+'_q_eval',
                                    #chkpt_dir=self.chkpt_dir
                                  )

        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.input_dims,
                                    #name=self.env_name+'_'+self.algo+'_q_next',
                                   # chkpt_dir=self.chkpt_dir
                                  )

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):        
         
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)        
        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec                            if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        print("nothing to save")

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()
        indices = np.arange(self.batch_size)
        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()


# In[316]:


agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                    input_dims=(env.state_space),
                    n_actions=env.action_space, mem_size=50000, eps_min=0.1,
                    batch_size=8, replace=1000, eps_dec=1e-5,
                    chkpt_dir='models/', algo='DQNAgent',
                    env_name='ENV-56')


# In[ ]:





# In[317]:


import gym
import numpy as np
#from dqn_agent import DQNAgent
#from utils import plot_learning_curve, make_env
from gym import wrappers

if __name__ == '__main__':
  #
    #env = gym.make('CartPole-v1')
    best_score = -np.inf
    load_checkpoint = False
    n_games = 1

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_'             + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    # if you want to record video of your agent playing, do a mkdir tmp && mkdir tmp/dqn-video
    # and uncomment the following 2 lines.
    #env = wrappers.Monitor(env, "tmp/dqn-video",
    #                    video_callable=lambda episode_id: True, force=True)
    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0
        stp = 0
        steps = []
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
           # print(observation_, reward, done, info)
            score += reward
            steps.append(action)
            #(int(next_state),reward, done, "extra")
            if not load_checkpoint:
                agent.store_transition(observation, action,
                                     reward, observation_, done)
                agent.learn()
            observation = observation_
            n_steps += 1
            stp = stp + 1
            if stp > 2500:
                break
        scores.append(score)
        steps_array.append(n_steps)
        print('Step taken to complete : ', stp)
        
        avg_score = np.mean(scores[-100:])
        print('episode: ', i,'score: ', score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)

    x = [i+1 for i in range(len(scores))]
   # plot_learning_curve(steps_array, scores, eps_history, figure_file)


# In[308]:


env.reset()


# In[309]:


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_classes, input_dims):
        super(DeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 4)
       
# m1: [32 x 56], m2: [1 x 64]
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        print('forward..')
        print(data)
        layer1 = F.relu(self.fc1(data))
        layer2 = self.fc2(layer1)        

        return layer2


# In[ ]:




