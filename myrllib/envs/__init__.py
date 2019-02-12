#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 15:52:14 2018

@author: qiutian
"""
from gym.envs.registration import register
from .simple_maze import Maze2DEnvV1
from .complex_maze import Maze2DEnvV2


register(
    'Maze2D-v1',
    entry_point='myrllib.envs.simple_maze:Maze2DEnvV1',
    max_episode_steps=100000
)

register(
    'Maze2D-v2',
    entry_point='myrllib.envs.complex_maze:Maze2DEnvV2',
    max_episode_steps=100000
)

