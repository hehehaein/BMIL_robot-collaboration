import gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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
    # 최적화 중에 다음 행동을 결정하기 위해서 하나의 요소 또는 배치를 이용해 호촐됩니다.
    # ([[left0exp,right0exp]...]) 를 반환합니다.
    def forward(self, x):
        m = nn.LeakyReLU(0.1)
        x = m(self.linear1(x))
        x = m(self.linear2(x))
        return self.linear3(x)
path = os.path.join(os.getcwd(), 'results')
policy_net = DQN().to(device)
target_net = DQN().to(device)
policy_net.load_state_dict(torch.load(path+'/200000_20_0.9_3.0625e-07_10_1_1th_100000_0.0001_1_True_False_False_0.95-19-8-4')['policy_net'])
policy_net.eval()

select_env = "dqn-v0"
register_env(select_env, lambda config: My_DQN())
env = gym.make(select_env).unwrapped
n_actions = env.action_space.n

throughputs = []
rewards = []
trajectory = []
throughput_count = 0
max_count = 0
stay = 0

def select_action(state):
    argmaxAction = policy_net(state).max(-1)[1].view(1, 1)
    return argmaxAction

# Set the environment and initial state

for t in range(0,20,1):
    # 행동 선택과 수행
    state = env.reset()
    # state = torch.Tensor(state)
    state = torch.Tensor([2., 0., 4., 0., 2., 3., 3., 3., 0., 0., 0., 2., 4., 4., 4., 0.])
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
    rewards.append(reward)
    reward = torch.tensor([reward], device=device)
    print('State:{}, Action:{}'.format(state, action))
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
    throughputs.append(throughput_count)
plt.figure()
plt.title('throughput')
plt.xlabel('episode')
plt.ylabel('throughput_count')
plt.plot(throughputs)

plt.figure()
plt.title('reward mean')
plt.xlabel('episode')
plt.ylabel('Reward')
plt.plot(rewards)

print(stay)

fig = plt.figure()
plt.figure()

plt.ioff()
plt.show()
