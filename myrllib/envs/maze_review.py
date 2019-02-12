#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 09:14:07 2018

@author: baobao
"""

import numpy as np 
import copy
import gym 
from gym import spaces 
from gym.utils import seeding 

class Maze2D(gym.Env):
    def __init__(self, width=18, height=23, action_dim=4, 
            r_goal=1.0, r_obs=-1.0, r_step=-0.01):
        super(Maze2D, self).__init__()
        ### env: m x n matrix, 0 or 1
        self._env = np.zeros((height, width), dtype=np.bool)
        self._start = (11, 16); self._goal = (1, 15)
        #self._start = (11,15); self._goal = (2,16)
        #self._start = (10,5); self._goal = (9, 1)
        self.width = width; self.height = height 
        self.r_goal = r_goal; self.r_obs = r_obs; self.r_step=r_step 

        self.observation_space = spaces.Discrete(width*height)
        self.action_space = spaces.Discrete(action_dim) 
        
        self._state = self.to_1d_state(self._start)
        self.generate_default_env_old()

    def reset(self):
        self._state = self.to_1d_state(self._start)
        return self._state 

    def to_1d_state(self, state_2d):
        return int(state_2d[0]*self.width + state_2d[1])

    def to_2d_state(self, state_1d):
        state_2d_0 = state_1d // self.width 
        state_2d_1 = state_1d - self.width*state_2d_0 
        return (int(state_2d_0), int(state_2d_1))
        
    
    def reset_env(self, env):
        self._env = copy.deepcopy(env)
        
    def step(self, action):
        assert self.action_space.contains(action)
        state_2d = self.to_2d_state(self._state)
        done = False; r = 0

        if action == 0: # up
            new_2d_0 = state_2d[0] - 1; new_2d_1 = state_2d[1]
        elif action == 1: # down
            new_2d_0 = state_2d[0] + 1; new_2d_1 = state_2d[1]
        elif action == 2: # left
            new_2d_0 = state_2d[0]; new_2d_1 = state_2d[1] - 1
        elif action == 3: #right
            new_2d_0 = state_2d[0]; new_2d_1 = state_2d[1] + 1
        else:
            print('Error! Unkown action...')

        if new_2d_0==self._goal[0] and new_2d_1==self._goal[1]:
            done = True
            r = self.r_goal 
        else: 
            if new_2d_0 < 0:
                new_2d_0 = 0; r = self.r_obs 
            if new_2d_0 > self.height - 1:
                new_2d_0 = self.height - 1; r = self.r_obs 
            if new_2d_1 < 0:
                new_2d_1 = 0; r = self.r_obs 
            if new_2d_1 > self.width - 1:
                new_2d_1 = self.width - 1; r = self.r_obs 

            if self._env[new_2d_0, new_2d_1]:
                new_2d_0 = state_2d[0]; new_2d_1 = state_2d[1]
                r = self.r_obs 
        
        r += self.r_step 
        self._state = self.to_1d_state((new_2d_0, new_2d_1))

        return self._state, r, done, None  
        
    def generate_default_env_old(self):
        env = copy.deepcopy(self._env)
        env[0]  = [1]*18
        env[1]  = [1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1]
        env[2]  = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        env[3]  = [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        env[4]  = [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1]
        env[5]  = [1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1]
        env[6]  = [1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,1,1,1]
        env[7]  = [1,1,1,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1]
        env[8]  = [1,1,0,0,0,0,0,0,1,0,0,0,1,1,1,1,1,1]
        env[9]  = [1,1,0,0,1,0,0,0,1,0,0,0,0,1,1,1,1,1]
        env[10] = [1,1,1,1,1,1,0,0,1,0,0,0,0,0,0,0,1,1]
        env[11] = [1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1]
        env[12] = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]
        env[13] = [1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1]
        env[14] = [1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1]
        env[15] = [1,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,1,1]
        env[16] = [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1]
        env[17] = [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1]
        env[18] = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1]
        env[19] = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]
        env[20] = [1,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1]
        env[21] = [1,0,0,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1]
        env[22] = [1]*18
        self._env = copy.deepcopy(env)
        
    def generate_default_env_new(self, old_env=None):
        if old_env is None:
            env = copy.deepcopy(self._env)
        else:
            env = copy.deepcopy(old_env)
        env[8,9:13] = 1
        self._env = copy.deepcopy(env)
    


    '''
    def generate_default_env_old(self):
        env = copy.deepcopy(self._env)
        env[0]  = [1]*12
        env[1]  = [1]*12
        env[2]  = [1,1,0,0,0,0,0,0,0,0,1,1]
        env[3]  = [1,1,0,0,0,0,0,0,0,0,1,1]
        env[4]  = [1,1,0,0,0,0,0,0,0,0,1,1]
        env[5]  = [1,1,0,0,0,0,0,0,0,0,1,1]
        env[6]  = [1,1,0,0,0,0,0,0,0,0,1,1]
        env[7]  = [1,1,0,0,0,0,0,0,0,0,1,1]
        env[8]  = [1,1,0,0,0,0,0,0,0,0,1,1]
        env[9]  = [1,0,0,1,0,0,0,0,1,0,1,1]
        env[10] = [1,0,1,1,1,0,0,1,1,1,1,1]
        env[11] = [1]*12
        self._env = copy.deepcopy(env)
        
    def generate_default_env_new(self, old_env=None):
        if old_env is None:
            env = copy.deepcopy(self._env)
        else:
            env = copy.deepcopy(old_env)
        #env[8,9:13] = 1
        env[7:9, 4] = 1
        self._env = copy.deepcopy(env)
    '''
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
    
