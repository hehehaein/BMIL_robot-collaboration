import copy
import pandas as pd
import gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns;
import networkx as nx
import time
import os
import math
from collections import namedtuple, deque
from itertools import count
from ray.tune.registry import register_env
from gymExample.gym_example.envs.my_dqn_env import My_DQN

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.ion()
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(16, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 81)
        """# Linear 입력의 연결 숫자는 conv2d 계층의 출력과 입력 이미지의 크기에
        # 따라 결정되기 때문에 따로 계산을 해야합니다.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)"""
    # 최적화 중에 다음 행동을 결정하기 위해서 하나의 요소 또는 배치를 이용해 호촐됩니다.
    # ([[left0exp,right0exp]...]) 를 반환합니다.
    def forward(self, x):
        # x = x.to(device)
        # print('forward\n',x)
        m = nn.LeakyReLU(0.1)
        x = m(self.linear1(x))
        x = m(self.linear2(x))
        return self.linear3(x)
path = os.path.join(os.getcwd(), 'results')
policy_net = DQN().to(device)
target_net = DQN().to(device)
policy_net.load_state_dict(torch.load(path+'/200000_20_0.9_4.9e-07_10_1_1th_100000_0.0001_1_True_False_1th_False-11-9-40')['policy_net'])
policy_net.eval()

select_env = "dqn-v0"
register_env(select_env, lambda config: My_DQN())
env = gym.make(select_env).unwrapped
n_actions = env.action_space.n


trajectory = []
def select_action(state):
    argmaxAction = policy_net(state).max(-1)[1].view(1, 1)
    return argmaxAction
# Set the environment and initial state
# state = env.reset()

scatter0 = []
scatter1 = []
scatter2 = []
scatter3 = []
scatter4 = []

iter_num = 1000
throughputs = np.zeros((iter_num,10))
rewards = np.zeros((iter_num,10))
for i in range(iter_num):
    state = env.reset()
    state = torch.Tensor(state)
    #state = torch.Tensor([2., 0., 4., 0., 2., 3., 3., 3., 0., 0., 0., 2., 4., 4., 4., 0.])
    max_count = 0
    stay = 0
    for t in range(0, 10, 1):
        # 행동 선택과 수행
        throughput_count = 0
        action = select_action(state)
        next_state, reward, done, last_set = env.step(action.item())
        state_reshape = np.reshape(state, (env.N + 2, 4))
        next_state_reshape = np.reshape(next_state, (env.N + 2, 4))
        if next_state_reshape[0][0] == 1 and \
                next_state_reshape[0][1] == 1 and \
                next_state_reshape[0][2] == 1 and \
                next_state_reshape[0][3] == 3:
            if np.array_equal(next_state_reshape, state_reshape):
                stay += 1
        rewards[i][t] = reward
        reward = torch.tensor([reward], device=device)
        print('State:{}, Reward:{}, throughput:{}, e_txr:{}'.format(state_reshape[0], reward.item(), last_set[0], last_set[4]))
        # To plot the trajectory, Storing the state information
        trajectory.append(np.array(state_reshape[0]))
        next_state = torch.Tensor(next_state)
        if done:
            next_state = None
        # Transition from current state to next state
        state = next_state
        if done:
            break
        # To plot the throughputs, Store the througput information
        if last_set[0] != 0:
            throughput_count += 1
        throughputs[i][t] = throughput_count
        '''if i == 0:
            scatter0.append(np.array(state_reshape[0]))
        elif i == 1:
            scatter1.append(np.array(state_reshape[0]))
        elif i == 2:
            scatter2.append(np.array(state_reshape[0]))
        elif i == 3:
            scatter3.append(np.array(state_reshape[0]))
        elif i == 4:
            scatter4.append(np.array(state_reshape[0]))'''

    print(stay)
    '''fig, ax1 = plt.subplots()
        color_1 = 'tab:blue'
        ax1.set_title('reward', fontsize=16)
        ax1.set_xlabel('steps')
        ax1.set_ylabel('reward', fontsize=14)
        ax1.plot(ranges, rewards, color='red')
        ax1.tick_params(axis='y')
        # right side with different scale
        ax2 = ax1.twinx()
        # instantiate a second axes that shares the same x-axis
        color_2 = 'tab:red'
        ax2.set_ylabel('throughput', fontsize=14)
        ax2.plot(ranges, throughputs, color='blue')
        ax2.tick_params(axis='y')
        fig.tight_layout()
        plt.show()'''

    '''plt.figure()
    plt.title('throughput')
    plt.xlabel('step')
    plt.ylabel('throughput')
    plt.plot(throughputs)

    plt.figure()
    plt.title('reward')
    plt.xlabel('step')
    plt.ylabel('Reward')
    plt.plot(rewards)'''

def make_list(episode, term):
    i = 0
    n = 0
    list = []
    while i < episode:
        for t in range(term):
            list.append(n)
            i += 1
        n += 1
    return list

plt.figure()
sns.set_style('darkgrid')
rewards = np.transpose(rewards).flatten()
throughputs = np.transpose(throughputs).flatten()
d = {'step': make_list(len(rewards), iter_num),
     'throughput': throughputs,
     'reward': rewards}
df = pd.DataFrame(data=d)
fig, axe1 = plt.subplots()
axe2 = axe1.twinx()
through = sns.lineplot(ax=axe1, data=df, x='step', y='throughput', color='red')
reward = sns.lineplot(ax=axe2, data=df, x='step', y='reward', color='blue')
axe1.legend(['throughput','reward'])


axe1.set_ylabel('throughput',fontsize=14)
axe2.set_ylabel('reward',fontsize=14)





# 3D 그래프 그리기
'''fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title('position')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d(0, env.MAX_LOC)
ax.set_ylim3d(env.MAX_LOC, 0)
ax.set_zlim3d(0, env.MAX_LOC)
color_list = ("olive", "orange", "green", "blue", "purple", "black", "cyan", "pink", "brown", "darkslategray")

nodes = []
nodes.append(env.source)
nodes.append(env.dest)
nodes.append(env.agent2)
ax.scatter(np.transpose(scatter0)[0], np.transpose(scatter0)[1], np.transpose(scatter0)[2],
           marker='o', s=60, c='green')
ax.scatter(np.transpose(scatter1)[0], np.transpose(scatter1)[1], np.transpose(scatter1)[2],
           marker='o', s=60, c='red')
ax.scatter(np.transpose(scatter2)[0], np.transpose(scatter2)[1], np.transpose(scatter2)[2], marker='o',
           s=60, c='orange')
ax.scatter(np.transpose(scatter3)[0], np.transpose(scatter2)[1], np.transpose(scatter2)[2], marker='o',
           s=60, c='purple')
ax.scatter(np.transpose(scatter4)[0], np.transpose(scatter2)[1], np.transpose(scatter2)[2], marker='o',
           s=60, c='blue')
ax.scatter(np.transpose(nodes)[0], np.transpose(nodes)[1], np.transpose(nodes)[2], marker='o', s=80, c='cyan')'''

plt.ioff()
plt.show()