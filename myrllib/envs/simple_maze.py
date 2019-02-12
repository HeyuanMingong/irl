#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the simple maze case in Section IV-B of the paper:
    [1] Zhi Wang, et al., "Incremental Reinforcement Learning with Prioritized Sweeping
        for Dynamic Environments", IEEE/ASME Transactions on Mechatronics.
"""

import numpy as np 
import copy
import gym 
from gym import spaces 
from gym.utils import seeding 
from myrllib.envs.maze import Maze2DEnv

class Maze2DEnvV1(Maze2DEnv):
    def __init__(self):
        super(Maze2DEnvV1, self).__init__(r_goal=1.0, r_obs=-1.0, r_step=-0.01,
                height=12, width=12, start=(10,5), goal=(9,1))
        self._state = self.to_1d_state(self._start)
        
    def set_env_old(self):
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
        env[9]  = [1,0,0,1,0,0,0,0,0,0,1,1]
        env[10] = [1,0,1,1,1,0,0,1,1,1,1,1]
        env[11] = [1]*12
        self._env = copy.deepcopy(env)
        
    def set_env_new(self):
        env = copy.deepcopy(self._env)
        env[0]  = [1]*12
        env[1]  = [1]*12
        env[2]  = [1,1,0,0,0,0,0,0,0,0,1,1]
        env[3]  = [1,1,0,0,0,0,0,0,0,0,1,1]
        env[4]  = [1,1,0,0,0,0,0,0,0,0,1,1]
        env[5]  = [1,1,0,0,0,0,0,0,0,0,1,1]
        env[6]  = [1,1,0,0,0,0,0,0,0,0,1,1]
        env[7]  = [1,1,0,0,1,0,0,0,0,0,1,1]
        env[8]  = [1,1,0,0,1,0,0,0,0,0,1,1]
        env[9]  = [1,0,0,1,0,0,0,0,0,0,1,1]
        env[10] = [1,0,1,1,1,0,0,1,1,1,1,1]
        env[11] = [1]*12
        self._env = copy.deepcopy(env)
    

    
