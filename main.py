#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the code for the paper:
[1] Zhi Wang, Chunlin Chen, Han-Xiong Li, Daoyi Dong, and Tzyh-Jong Tarn, 
    "Incremental Reinforcement Learning with Prioritized Sweeping for Dynamic 
    Environments", IEEE/ASME Transactions on Mechatronics, 2019.

The implementation consists of four steps:
1. Train an agent with Q-learning in an original environment
2. In a new environment, executing a virtual learning process to detect 
    the drift environment
3. Execute the prioritized sweeping process of the drift environment and 
    its neighbor environment
4. Start a new learning process till convergence 

https://github.com/HeyuanMingong/irl.git
"""
### common lib
import sys
import gym
import numpy as np
import argparse 
from tqdm import tqdm
import os
import time 
start_time = time.time()
import scipy.io as sio
import copy 

### personal lib 
from myrllib.algorithms.qlearning import QLearning 
import myrllib.envs

######################## Arguments ############################################
parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99, 
        help='discount factor in Q-learning')
parser.add_argument('--lr', type=float, default=1e-2, 
        help='learning rate in Q-learning')
parser.add_argument('--max_steps', type=int, default=100000,
        help='max steps in one episode of the learning process')
parser.add_argument('--max_epochs', type=int, default=1000,
        help='max learning episodes')
parser.add_argument('--output', type=str, default='output',
        help='output folder for saving the results')
parser.add_argument('--pretrain', dest='pretrain', action='store_true',
        help='whether to pretrain in the original environment')
parser.add_argument('--no-pretrain', dest='pretrain', action='store_false')
parser.set_defaults(pretrain=True)
parser.add_argument('--finetune', dest='finetune', action='store_true',
        help='whether to finetune in the new environment')
parser.add_argument('--no-finetune', dest='finetune', action='store_false')
parser.set_defaults(finetune=True)
parser.add_argument('--ran', dest='ran', action='store_true',
        help='the baseline of RL without pi_old, learning from scratch')
parser.add_argument('--no-ran', dest='ran', action='store_false')
parser.set_defaults(ran=True)
parser.add_argument('--fine', dest='fine', action='store_true',
        help='the baseline if RL with pi_old, learning based on existing knowledge')
parser.add_argument('--no-fine', dest='fine', action='store_false')
parser.set_defaults(fine=True)
parser.add_argument('--prq', dest='prq', action='store_true',
        help='baseline of PRQ-learning')
parser.add_argument('--no-prq', dest='prq', action='store_false')
parser.set_defaults(prq=True)
parser.add_argument('--incre', dest='incre', action='store_true',
        help='the proposed method, incremental reinforcement learning')
parser.add_argument('--no-incre', dest='incre', action='store_false')
parser.set_defaults(incre=True)
parser.add_argument('--trial', type=int, default=1)
parser.add_argument('--strategy', type=str, default='epsilon',
        help='exploration strategy, epsilon-greedy or softmax')
parser.add_argument('--env', type=str, default='Maze2D-v1',
        help='maze environemnt, simple maze or complex maze')
parser.add_argument('--incre_m', type=int, default=1,
        help='m-degree neighboring environment')
parser.add_argument('--ps_iter', type=int, default=300,
        help='iteraion number of the prioritized sweeping process')
parser.add_argument('--ps_lr',type=float, default=1.0,
        help='learning rate of the dynamic programming process')
parser.add_argument('--nu', type=float, default=0.99,
        help='hyperparameter of PRQ-learning')
args = parser.parse_args()
print(args)
np.random.seed(args.trial)


env = gym.make(args.env).unwrapped
### hyperparameters of the exploration strategies
if args.env == 'Maze2D-v1':
    epsilon_pre = list(np.linspace(0.1, 0.0, 900)) + [0]*100
    epsilon_ran = epsilon_pre
    epsilon_fine = epsilon_pre
    epsilon_prq = epsilon_pre
    epsilon_incre = epsilon_pre 

    tau_pre = list(np.linspace(10,50,500)) + [50]*500
    tau_ran = tau_pre
    tau_fine = tau_pre
    tau_prq = tau_pre
    tau_incre = tau_pre

elif args.env == 'Maze2D-v2':
    epsilon_pre = [1.0]*300 + list(np.linspace(1.0, 0.0, 600)) + [0.0]*4100
    epsilon_incre = list(np.linspace(0.1, 0.0, 900)) + [0]*4100
    epsilon_ran = epsilon_pre
    epsilon_fine = list(np.linspace(1.0, 0.1, 100)) + list(np.linspace(
        0.1, 0.0, 800)) + [0]*4100
    epsilon_prq = epsilon_pre

    tau_pre = list(np.linspace(10,50,1000)) + [50]*4000
    tau_prq = tau_pre
    tau_fine = tau_pre
    tau_ran = tau_pre
    tau_incre = tau_pre

### learning in the original environment
if args.pretrain:
    ### set the original environment
    ### see instructions in the file 'myrllib/envs/*'
    env.set_env_old() 
    learner = QLearning(env, gamma=args.gamma, lr=args.lr)
    steps_hist = np.zeros(args.max_epochs, dtype=np.int32)
    data = {}
    for epoch in tqdm(range(args.max_epochs)):
        s = env.reset()
        for step in range(args.max_steps):
            a = learner.pi(s, epsilon=epsilon_pre[epoch], 
                    tau=tau_pre[epoch], strategy=args.strategy)
            s_next, r, done, _ = env.step(a)
            learner.step(s, a, r, s_next)
            if done:
                #a = env.action_space.sample()
                #learner.step(s_next, a, r, s_next)
                break 
            s = s_next 
        steps_hist[epoch] = step 
    data['steps_hist'] = steps_hist
    data['Q'] = learner.Q; data['R'] = learner.R 
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    name = os.path.join(args.output, 'pretrained.mat')
    print('Save the original environment to file: %s'%name)
    sio.savemat(name, data)

### learning in the new environment
if args.finetune:
    name = os.path.join(args.output, 'pretrained.mat')
    assert os.path.exists(name)
        
    print('Load the Q-value and R-value from the original environment: %s'%name)
    pretrained = sio.loadmat(name)
    pre_Q = pretrained['Q']; pre_R = pretrained['R']

    ### set the new environment
    ### see instructions in 'myrllib/envs/*'
    env.set_env_new()

    ### a single learning process of baselines 1&2, and IRL 
    def learning_process(learner, epsilon=[1.0]*10000,
            tau=[1.0]*10000, strategy='epsilon'):
        steps = np.zeros(args.max_epochs, dtype=np.int32)
        for epoch in tqdm(range(args.max_epochs)):
            s = env.reset()
            for step in range(args.max_steps):
                a = learner.pi(s, epsilon=epsilon[epoch], tau=tau[epoch],
                        strategy=strategy)
                s_next, r, done, _ = env.step(a)
                learner.step(s, a, r, s_next)
                if done:
                    break 
                s = s_next 
            steps[epoch] = step 
        return steps 

    ### a virtual learning process for drift detection
    ### only record the reward function, no value iteration
    def virtual_learning(learner):
        for epoch in range(10):
            s = env.reset()
            for step in range(args.max_steps):
                a = learner.pi(s, epsilon=1.0)
                s_next, r, done, _ = env.step(a)
                learner.virtual_step(s, a, r, s_next)
                if done:
                    break 
                s = s_next 

    ### baseline 1:  'RL without pi_old', i.e., learning from scratch
    if args.ran:
        learner_ran = QLearning(env, gamma=args.gamma, lr=args.lr)
        steps_ran = learning_process(learner_ran, epsilon=epsilon_ran, 
                tau=tau_ran, strategy=args.strategy)
        
        name = os.path.join(args.output, 'ran_%d.npy'%args.trial)
        print('Save baseline <RL without pi_old> to: %s'%name)
        np.save(name, steps_ran) 
        
    ### baseline 2:  of 'RL with pi_old', i.e., directly learning based on existing knowledge
    if args.fine:
        learner_fine = QLearning(env, gamma=args.gamma, lr=args.lr, Q=pre_Q)
        steps_fine = learning_process(learner_fine, epsilon=epsilon_fine,
                tau=tau_fine, strategy=args.strategy)

        name = os.path.join(args.output, 'fine_%d.npy'%args.trial)
        print('Save baseline <RL with pi_old> to: %s'%name)
        np.save(name, steps_fine) 

    ### baseline 3: 'PRQ-learning', a kind of policy transfer algorithm
    if args.prq:
        '''
        Hyperparameters of PRQ-learning, more details can be found in: 
        [2] Fernando Fernandez, Javier Garcia, and Manuela Veloso, 
            "Probabilistic Policy Reuse for inter-task transfer learning", 
            Robotics and Autonomous Systems, 2010.
        '''
        ### upsilon: temperature for weighting the old and new policies
        ### psi: the probability for using the old policy
        ### nu: weight decay for using the old policy
        upsilon = 1; nu = args.nu; psi = 1.0
        score_old, score_new = 0.0, 0.0; used_old, used_new = 0, 0
        ### record the probability for selecting the old policy
        ### for debuging the PRQ-learning algorithm
        select_old_p = []

        learner_prq = QLearning(env, gamma=args.gamma, lr=args.lr, Q_reuse=pre_Q)
        steps_prq = np.zeros(args.max_epochs, dtype=np.int32)
        for epoch in tqdm(range(args.max_epochs)):
            s = env.reset()
            for step in range(args.max_steps):
                ### select old policy or new policy according to their scores
                p_old = np.exp(upsilon * score_old) / (np.exp(upsilon * score_old) 
                        + np.exp(upsilon * score_new))
                select_old = np.random.binomial(n=1, p=p_old, size=1)
                select_old_p.append(p_old*psi)

                to_use_new = True
                if select_old == 1:
                    reuse = np.random.binomial(n=1, p=psi, size=1)
                    ### reuse the old policy
                    if reuse == 1:
                        #print('Use old policy...')
                        a = learner_prq.pi_reuse(s)
                        s_next, r, done, _ = env.step(a)
                        score_old = (score_old * used_old + r) / (used_old + 1)
                        used_old += 1
                        to_use_new = False

                ### use the new policy in the new environment
                if to_use_new:
                    a = learner_prq.pi(s, epsilon=epsilon_prq[epoch], 
                            tau=tau_prq[epoch], strategy=args.strategy)
                    s_next, r, done, _ = env.step(a)
                    score_new = (score_new * used_new + r) / (used_new + 1)
                    used_new += 1

                learner_prq.step(s, a, r, s_next)
                if done:
                    break 
                s = s_next

            ### decay the probability for reusing the old policy 
            psi *= nu
            steps_prq[epoch] = step  

        name = os.path.join(args.output, 'prq_%d.npy'%args.trial)
        print('Save baseline <PRQ-learning> to file: %s'%name)
        np.save(name, steps_prq)
        np.save(os.path.join(args.output, 'prq_info_%d'%args.trial), 
                np.array(select_old_p))

    ### the proposed method, incremental reinforcement learning
    if args.incre:
        learner_incre = QLearning(env, gamma=args.gamma, lr=args.lr, Q=pre_Q)

        ### Detection of the drift environment
        print('Execute drift detection...')
        virtual_learning(learner_incre)
        drift_env, drift_env_2d = learner_incre.drift_detection(pre_R)
        print('The drift environment is: ', drift_env_2d)

        ### prioritized sweeping of drift environment
        print('Execute the prioritized sweeping process over %d-degree '
                'neighboring environment'%args.incre_m)
        learner_incre.prioritized_sweeping(drift_env, 
                m=args.incre_m, lr=args.ps_lr, max_iters=args.ps_iter)
        steps_incre = learning_process(learner_incre, epsilon=epsilon_incre,
                tau=tau_incre, strategy=args.strategy)

        name = os.path.join(args.output, 'incre_%d.npy'%args.trial)
        print('Save the proposed method <IRL> to file: %s'%name)
        np.save(name, steps_incre)





print('Running time: %.2f'%(time.time()-start_time))


