# Incremental reinforcement learning with prioritized sweeping for dynamic environments

This repo contains code accompaning the paper: [Zhi Wang, Chunlin Chen, Han-Xiong Li, Daoyi Dong, and Tzyh-Jong Tarn, "Incremental reinforcement learning with prioritized sweeping for dynamic environments", *IEEE/ASME Transactions on Mechatronics*, 2019.](http://heyuanmingong.github.io/data/IRL/IRL.pdf)
It contains code for running the incremental learning tasks with a discrete state-action space, including the simple maze and complex maze domains, as stated in the paper. Click [here](https://heyuanmingong.github.io#irl) to get more details about this work.

### Dependencies
This code requires the following:
* python 3.\*
* gym

### Data
* For the simple maze domain, data is generated from `myrllib/envs/simple_maze.py`
* For the complex maze domain, data is generated from `myrllib/envs/complex_maze.py`

### Usage 
* For example, to run the code in the simple maze domain with epsilon-greedy strategy, just run the bash script `./simple_maze_epsilon.sh`, also see the usage instructions in the script and `main.py`
* When getting the results in the folder `output/*`, plot the results using `data_process.py`. For example, the results for `./simple_maze_epsilon.sh` is as follow:
![experimental results for simple maze with epsilon-greedy strategy](https://github.com/HeyuanMingong/irl/blob/master/exp/maze_simple_epsilon.png)

Also, the results for other bash scripts are shown in `exp/*`

### Contact 
For safety reasons, the code is coming soon.

To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/HeyuanMingong/irl/issues), or email to njuwangzhi@gmail.com.
