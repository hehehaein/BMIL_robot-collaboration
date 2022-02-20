#!/usr/bin/env python
# encoding: utf-8

from gymExample.gym_example.envs.example_env import Example_v0
from gymExample.gym_example.envs.my_ppo_env import My_PPO
from gymExample.gym_example.envs.my_dqn_env import My_DQN

from ray.tune.registry import register_env
import gym
import os
import ray
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.dqn as dqn

import numpy as np
from matplotlib import pyplot as plt

import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_STEPS = 2000
# ~number of relay node
N = 2
# ~transmission radius max
R_MAX = 5
# location x,y,z
MIN_LOC = 0
MAX_LOC = 4

MAX_HEIGHT = 3
MIN_HEIGHT = 0

source = np.array((MIN_LOC, MIN_LOC, MIN_HEIGHT, R_MAX))
dest = np.array((MAX_LOC, MAX_LOC, MAX_HEIGHT, R_MAX))
agent2 = np.array((3,3,3,3))

def main ():
    # init directory in which to save checkpoints
    chkpt_root = "tmp/exa"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)


    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True)

    #register the custom environment
    select_env = "mydqn-v0"
    register_env(select_env, lambda config: My_DQN())
    #configure the environment and create agent
    config = dqn.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["framework"] = "torch"
    config["num_workers"] = 1
    config['exploration_config']['final_epsilon'] = 0.04
    config['exploration_config']['epsilon_timesteps'] = 10000
    config["gamma"] = 0.99
    config["lr"] = 0.04
    config["dueling"] = False
    config["double_q"] = False


    agent = dqn.DQNTrainer(config, env=select_env)

    """#register the custom environment
    select_env = "myppo-v0"
    register_env(select_env, lambda config: My_PPO())
    #configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["framework"] = "torch"
    config["num_workers"] = 1
    agent = ppo.PPOTrainer(config, env=select_env)"""


    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} saved {}"
    n_iter = 1

    # train a policy with RLlib using PPO/DQN
    for n in range(n_iter):
        result = agent.train()
        chkpt_file = agent.save(chkpt_root)

        print(status.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                chkpt_file
                ))

    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    # print(model.base_model.summary())

    # apply the trained policy in a rollout
    agent.restore(chkpt_file)
    env = gym.make(select_env)

    state = env.reset()
    sum_reward = 0
    n_step = 20
    rewards = np.zeros(n_step*n_iter)
    i=0

    weights = agent.get_weights()

    for key, val in weights.items():
        weights_val = val

    wei_vlist = []
    for i in weights_val.items():
        wei_vlist.append(i)

    V_wei = []
    V_bias = []
    for i in range(0, 8):
        if i != 4 and i != 5:
            if i % 2 == 0:
                V_wei.append(wei_vlist[i])
            else:
                V_bias.append(wei_vlist[i])

    for step in range(n_step):
        action = agent.compute_single_action(state)
        state, reward, done, info = env.step(action)
        #print(action, state)

        out = F.relu(F.linear(torch.from_numpy(state).float(), torch.from_numpy(V_wei[0][1]),
                              torch.from_numpy(V_bias[0][1])))
        out = F.relu(F.linear(out, torch.from_numpy(V_wei[1][1]),
                              torch.from_numpy(V_bias[1][1])))
        #Q = F.linear(out, torch.from_numpy(V_wei[2][1]),
                                     #torch.from_numpy(V_bias[2][1]))

        print(out)

        #rewards[i] = reward
        i = i+1
        if sum_reward < reward:
            sum_reward=reward
            high_state = state.copy()
        env.render()

        if done:
            # report at the end of each episode
            #print("cumulative reward", sum_reward)
            print("highest reward ", sum_reward)
            print("state ", high_state)
            state= env.reset()
            sum_reward = 0

    plt.figure()
    #plt.plot(rewards)

    # node들의 좌표 배열 구하기
    scatter_array = np.empty((4, N+2), int)
    scatter_array[0] = high_state.copy()
    scatter_array[1] = agent2.copy()
    scatter_array[2] = source.copy()
    scatter_array[3] = dest.copy()

    def create_sphere(cx, cy, cz, r):

        u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]

        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        # shift and scale sphere
        x = r * x + cx
        y = r * y + cy
        z = r * z + cz
        return (x, y, z)

    #3D 그래프 그리기
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim3d(0 - 2, MAX_LOC + 2)
    ax.set_ylim3d(MAX_LOC + 2, 0 - 2)
    ax.set_zlim3d(0 - 2, MAX_LOC + 2)

    color_list = ("red","orange","green","blue","purple","black")

    for i in range(0,N+2,1):
        #구 그리기
        (x,y,z) = create_sphere(scatter_array[i][0], scatter_array[i][1],scatter_array[i][2],scatter_array[i][3])
        ax.auto_scale_xyz([0, 500], [0, 500], [0, 0.15])
        ax.plot_surface(x,y,z,color=color_list[i],linewidth=0,alpha=0.3)
        # 점 찍기
        ax.scatter(np.transpose(scatter_array)[0],np.transpose(scatter_array)[1],np.transpose(scatter_array)[2], marker='o', s=80, c='darkgreen')

    plt.show()

if __name__ == "__main__":
    main()