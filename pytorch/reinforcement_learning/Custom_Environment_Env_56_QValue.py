#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports
import numpy as np


# In[10]:


class Env56:
    def __init__(self, final_state_val):
        print('My Env Created')
        self.action_space = 4 # up and down , left , right 0, 1, 2, 3
        self.state_space = 56 # 
        self.current_state = int(0)
        self.q_val = np.zeros([self.state_space, self.action_space])
        self.final_state = final_state_val
    def reset(self):
        #print('reset done')
        self.current_state = 0
        return self.current_state
        #step will take action as input and will return reward, next state, done flag means is action lead to final
        # state
    def getNextState(self, action):
        
        # print('current state')
        #print(self.current_state)
       # print(action)
               
        if action == 0 and int(env.current_state) in [i for i in range(8)]:
           # print('first check')
            return self.current_state
        if action == 1 and int(env.current_state) in [i for i in range(48, 56)]:
          #  print('2 check')
            return self.current_state
        if action == 2 and int(env.current_state) == 0:
           # print('3 check')
            return self.current_state
        if action == 3 and int(env.current_state) == 55:
          #  print('4 check')
            return sel.current_state
        if action == 2:
          #  print('5 check')
            next_ = int(self.current_state) - 1
        if action == 3:
          #  print('6 check')
            next_ = int(self.current_state) + 1
        if action == 0:
          #  print('7 check')
            next_ = int(self.current_state) - 8
        if action == 1:
          #  print('8 check')
            next_ = int(self.current_state) + 8
            
        
       # print(next_)
        if next_ > 55 or next_ < 0:
            print(self.current_state)
            print(action)
            
        return next_
            
    def step(self, action):
        next_state = self.getNextState(action)
        #print('step')
      #  print(next_state)
        
        self.current_state = str(next_state)
        #print(self.current_state)
        done = (int(self.current_state) == int(self.final_state))
        reward = 0
        if done:
            reward = 1
        return (int(next_state), done, reward)


# In[11]:


class Action_Taker:
    def __init__(self, e, e_min, state_space, action_space):
        #print('action taker created')
        self.e = e
        self.e_min = e_min
        self.state_space = state_space
        self.action_space = action_space
        self.q_val = np.zeros([self.state_space, self.action_space])
    def act(self, state, e):
        if np.random.rand() < e:
            return np.random.choice(range(self.action_space))
       
        return np.argmax(self.q_val[state])


# In[12]:


env = Env56(27)


# In[13]:


action_taker = Action_Taker(0.991, 0.1, env.state_space, env.action_space)


# In[14]:


e = 0.9

for i in range(30):
    
    done = False
    count = 0
    state = env.reset()
    actions =[]
    discount = 0.98
   # 
    while done == False:
        action = action_taker.act(state, e)
        actions.append(action)
       # print(action)
        next_state, done, reward = env.step(action)
       # print(next_state, done, reward)
        #if done:
           # print('true is done',next_state)
        e = e * e
        if e < 0.1:
            e =0.1
        #print(next_state)
        action_taker.q_val[state, action] = action_taker.q_val[state, action] +  0.5 * (reward + discount * np.amax(action_taker.q_val[next_state]) - action_taker.q_val[state, action])
        if count > 500000:
            print(env.current_state)
            break
        count = count + 1
        state = next_state
    print(count)
    #print(actions)


# In[9]:


actions


# In[ ]:




