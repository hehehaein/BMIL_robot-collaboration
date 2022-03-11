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
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)

policy_net = DQN().to(device)
target_net = DQN().to(device)
policy_net.load_state_dict(torch.load('./test32000-0.8-1e-4_1th_txr2_explo42000_1_22_5_34')['policy_net'])

select_env = "dqn-v0"
register_env(select_env, lambda config: My_DQN())
env = gym.make(select_env).unwrapped
n_actions = env.action_space.n

throughputs = []
rewards = []
trajectory = []
def select_action(state):
    argmaxAction = policy_net(state).max(-1)[1].view(1, 1)
    return argmaxAction
# Set the environment and initial state
state = env.reset()
state = torch.Tensor(state)
throughput_count = 0
max_count = 0
for t in count():
    # 행동 선택과 수행
    action = select_action(state)
    next_state, reward, done, last_set = env.step(action.item())
    state_reshape = np.reshape(state, (env.N + 2, 4))
    next_state_reshape = np.reshape(next_state, (env.N + 2, 4))
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
# reward_mean_list = []
# for i in range(int(len(rewards))):
#     a = rewards[i * env.MAX_STEPS + 1: i * env.MAX_STEPS + 1 + env.MAX_STEPS]
#     reward_mean_list.append(np.mean(a))
# reward_means = rewards
plt.figure()
plt.title('reward mean')
plt.xlabel('episode')
plt.ylabel('Reward')
plt.plot(rewards)
fig = plt.figure()
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
ax.scatter(np.transpose(trajectory)[0], np.transpose(trajectory)[1],
           np.transpose(trajectory)[2], marker='o',s=60, c='purple')
ax.scatter(np.transpose(nodes)[0], np.transpose(nodes)[1],
           np.transpose(nodes)[2], marker='o', s=80, c='cyan')
plt.ioff()
plt.show()
