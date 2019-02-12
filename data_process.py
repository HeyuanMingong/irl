#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 14:27:58 2018

@author: qiutian
"""

import numpy as np
import os
from myrllib.utils.myplot import simple_plot
import scipy.io as sio
import matplotlib.pyplot as plt

def arr_ave(arr, bs=1):
    arr = arr.squeeze()
    nl = arr.shape[0]//bs
    arr_n = np.zeros(nl)
    for i in range(nl):
        arr_n[i] = np.mean(arr[bs*i:bs*(i+1)])
    return arr_n


###############################################################################
r_path = 'output/maze_simple/epsilon'

### plot the results in the original environment
def pretrain():
    cutoff = 1000; bs = 1
    data = sio.loadmat(os.path.join(r_path, 'pretrained.mat'))
    steps = arr_ave(data['steps_hist'].squeeze()[:cutoff], bs=bs)
    simple_plot([steps])
    return data
data = pretrain()    


###############################################################################
### plot the results in the new environment
def finetune():
    trial = 1
    f_ran = np.load(os.path.join(r_path, 'ran_%d.npy'%trial))
    f_fine = np.load(os.path.join(r_path, 'fine_%d.npy'%trial))
    f_prq = np.load(os.path.join(r_path, 'prq_%d.npy'%trial))
    f_incre = np.load(os.path.join(r_path, 'incre_%d.npy'%trial))
    
    cutoff = 1000; num = 100; bs = cutoff//num
    steps_ran = arr_ave(f_ran.reshape(-1)[:cutoff], bs=bs) 
    steps_fine = arr_ave(f_fine.reshape(-1)[:cutoff], bs=bs) 
    steps_prq = arr_ave(f_prq.reshape(-1)[:cutoff], bs=bs) 
    steps_incre = arr_ave(f_incre.reshape(-1)[:cutoff], bs=bs) 

    gap = 1
    xx = np.arange(0, num, gap); mark = num // (gap*10)
    plt.figure(figsize=(6,4)); plt.yscale('log')
    plt.plot(xx, steps_ran[xx], color='black', lw=2,
             marker='o', markevery=mark, ms=8, mew=2, mfc='white')
    plt.plot(xx, steps_fine[xx], color='green', lw=2,
             marker='s', markevery=mark, ms=8, mew=2, mfc='white')
    plt.plot(xx, steps_prq[xx], color='blue', lw=2,
             marker='^', markevery=mark, ms=8, mew=2, mfc='white')
    plt.plot(xx, steps_incre[xx], color='red', lw=2, 
             marker='x', markevery=mark, ms=8, mew=2, mfc='white')
    
    plt.legend(['RL without $\pi^*_{old}$', 'RL with $\pi^*_{old}$', 
                'PRQ-learning', 'IRL'],
               #bbox_to_anchor = (1, 0.3),
               fancybox=True, shadow=True, fontsize=15)
    plt.xlabel('Learning episodes', fontsize=18)
    plt.ylabel('Steps to the goal', fontsize=18)
    plt.xticks(np.arange(0,num+1,num//5), bs*np.arange(0,num+1,num//5),
               fontsize=10)
    plt.grid(axis='y', ls='--')
    
    return (f_ran, f_fine, f_prq, f_incre)
data = finetune()   





































