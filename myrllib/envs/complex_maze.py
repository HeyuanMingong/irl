#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the complex maze case in Section IV-C of the paper:
    [1] Zhi Wang, et al., "Incremental Reinforcement Learning with Prioritized Sweeping
        for Dynamic Environments", IEEE/ASME Transactions on Mechatronics, 2018.
"""

import numpy as np 
import copy
import gym 
from gym import spaces 
from gym.utils import seeding 
from myrllib.envs.maze import Maze2DEnv

class Maze2DEnvV2(Maze2DEnv):
    def __init__(self):
        super(Maze2DEnvV2, self).__init__(
                r_goal=1.0, r_obs=-1.0, r_step=-0.001,
                height=22, width=22, start=(20,1), goal=(1,19))
        self._state = self.to_1d_state(self._start)
        
    def set_env_old(self):
        env = copy.deepcopy(self._env)
        env[0]  = [1]*22
        env[1]  = [1,0,0,0,0,0,0,0,1,1,1,0,0,1,0,0,0,1,1,0,0,1]
        env[2]  = [1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1]
        env[3]  = [1,0,0,0,1,1,1,0,1,0,0,0,0,1,1,0,1,1,0,0,0,1]
        env[4]  = [1,1,0,0,0,1,1,1,0,1,0,1,0,0,1,0,0,0,0,0,0,1]
        env[5]  = [1,1,0,0,0,0,0,0,0,1,0,1,0,0,1,1,1,1,1,0,1,1]
        env[6]  = [1,1,1,1,1,0,1,1,0,1,0,0,1,0,0,0,0,0,0,0,1,1]
        env[7]  = [1,0,1,1,0,0,0,1,0,0,0,0,1,1,1,1,1,1,0,0,1,1]
        env[8]  = [1,0,0,1,0,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,1]
        env[9]  = [1,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,1,1,1]
        env[10] = [1,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1]
        env[11] = [1,0,0,1,1,0,0,0,1,0,0,1,0,0,0,0,1,1,1,0,1,1]
        env[12] = [1,0,0,0,1,1,0,0,1,1,0,1,0,1,0,0,0,0,1,0,0,1]
        env[13] = [1,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,1,1,1,0,1]
        env[14] = [1,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,1,0,1]
        env[15] = [1,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,1]
        env[16] = [1,0,0,1,0,0,1,1,1,0,0,1,0,0,1,0,0,1,0,0,0,1]
        env[17] = [1,0,0,0,0,0,1,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1]
        env[18] = [1,0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,1]
        env[19] = [1,0,1,0,0,0,0,0,0,1,0,1,0,1,1,0,0,1,1,0,0,1]
        env[20] = [1,0,0,0,1,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1]
        env[21] = [1]*22
        self._env = copy.deepcopy(env)
        
    def set_env_new(self):
        env = copy.deepcopy(self._env)
        env[0]  = [1]*22
        env[1]  = [1,0,0,0,0,0,0,0,1,1,1,0,0,1,0,0,0,1,1,0,0,1]
        env[2]  = [1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,1]
        env[3]  = [1,0,0,0,1,1,1,0,1,0,0,0,0,1,1,0,1,1,0,0,0,1]
        env[4]  = [1,1,0,0,0,1,1,1,0,1,0,1,0,0,1,0,0,0,0,0,0,1]
        env[5]  = [1,1,0,0,0,0,0,0,0,1,0,1,0,0,1,1,1,1,1,0,1,1]
        env[6]  = [1,1,1,1,1,0,1,1,0,1,0,0,1,0,0,0,0,0,0,0,1,1]
        env[7]  = [1,0,1,1,0,0,0,1,0,0,0,0,1,1,1,1,1,1,0,0,1,1]
        env[8]  = [1,0,0,1,0,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,1]
        env[9]  = [1,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,1,1,1]
        env[10] = [1,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,1,1]
        env[11] = [1,0,0,1,1,0,0,0,1,0,0,1,0,0,0,0,1,1,1,0,1,1]
        env[12] = [1,0,0,0,1,1,0,0,1,1,0,1,0,1,0,0,0,0,1,0,0,1]
        env[13] = [1,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,1,1,1,0,1]
        env[14] = [1,0,0,0,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,1,0,1]
        env[15] = [1,0,1,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,1]
        env[16] = [1,0,0,1,0,0,1,1,1,0,0,1,0,0,1,0,0,1,0,0,0,1]
        env[17] = [1,0,0,0,0,0,1,0,1,1,0,0,0,1,1,0,0,0,1,0,1,1]
        env[18] = [1,0,0,1,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,1]
        env[19] = [1,0,1,0,0,0,0,0,0,1,0,1,0,1,1,0,0,1,1,0,0,1]
        env[20] = [1,0,0,0,1,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1]
        env[21] = [1]*22

        env[13, 10] = 1

        self._env = copy.deepcopy(env)
    


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
    
