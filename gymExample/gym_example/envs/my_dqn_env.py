import copy
import os
from itertools import product

import gym
import numpy as np
import random
from gym.utils import seeding
import math
import networkx as nx
from matplotlib import pyplot as plt
from ray.rllib.train import torch

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
os.environ["PYTHONHASHSEED"]=str(seed)

class reward_set:
    def __init__(self, N):
        self.N = N

    # S-D까지 연결되는지 확인하기위해서 인접행렬 만들기
    def cal_adjacency(self, next_state_array):
        adj_array = np.empty((self.N + 2, self.N + 2), float)
        for i in range(0, self.N + 2, 1):
            for j in range(0, self.N + 2, 1):
                distance = math.sqrt(((next_state_array[i][0] - next_state_array[j][0]) ** 2)
                                     + ((next_state_array[i][1] - next_state_array[j][1]) ** 2)
                                     + ((next_state_array[i][2] - next_state_array[j][2]) ** 2))
                if distance <= next_state_array[i][3]:
                    adj_array[i][j] = distance
                else:
                    adj_array[i][j] = 0
        # print('adj\n',adj_array)
        return adj_array

    # 인접그래프 그리기.
    # S-D까지의 경로가 있나 확인 -> 홉의 개수(has_path최단거리)찾기 -> throughput 구하기
    def cal_throughput(self, adj_array):
        tmp_array = np.zeros((self.N + 2, 4))
        tmp2_array = np.zeros((self.N + 2, 4))
        tmp_array[0] = adj_array[2]
        tmp_array[1] = adj_array[0]
        tmp_array[2] = adj_array[1]
        tmp_array[3] = adj_array[3]
        tmp_array = np.transpose(tmp_array)
        tmp2_array[0] = tmp_array[2]
        tmp2_array[1] = tmp_array[0]
        tmp2_array[2] = tmp_array[1]
        tmp2_array[3] = tmp_array[3]
        tmp2_array = np.transpose(tmp2_array)
        graph = nx.Graph()
        for i in range(0, self.N + 2, 1):
            graph.add_node(i)
        for i in range(0, self.N + 2, 1):
            for j in range(i, self.N + 2, 1):
                if 0 < tmp2_array[i][j]:
                    graph.add_edge(i, j)

        if nx.has_path(graph, 0, self.N + 1):
            path_hop = self.N + 1
        else:
            path_hop = np.inf
        #print('tmp_array\n', tmp2_array)
        # print("path_hop : ",path_hop)
        if path_hop != np.inf:
            throughput = 20 / path_hop
        else:
            throughput = 0

        return throughput


    def cal_h(self, x, y, z, source, destination):
        kx = destination[0] - source[0]
        ky = destination[1] - source[1]
        kz = destination[2] - source[2]
        constant = (((kx * x) + (ky * y) + (kz * z)) / (math.pow(kx, 2) + math.pow(ky, 2) + math.pow(kz, 2)))
        h = math.sqrt(math.pow(constant * kx - x, 2) + math.pow(constant * ky - y, 2) + math.pow(constant * kz - z, 2))
        return h

    # (시간t일때의 수선의 발 - 시간t+1일때 수선의 발)길이 구하기
    def cal_foot_of_perpendicular(self, state_array, next_state_array, source, destination, i):
        foot_of_perpendicular = self.cal_h(state_array[i][0], state_array[i][1], state_array[i][2], source, destination) \
                                - self.cal_h(next_state_array[i][0], next_state_array[i][1], next_state_array[i][2],
                                             source, destination) \
                                - state_array[i][3] + next_state_array[i][3]
        return foot_of_perpendicular
    def cal_foot(self, next_state_array, source, destination, i):
        foot = next_state_array[i][3] \
               - self.cal_h(next_state_array[i][0], next_state_array[i][1], next_state_array[i][2], source, destination)
        return foot

    def cal_dispersed(self, i, my_txr, adj_array):
        adj_nodes = 0
        now_disperse = 0
        for j in range(0, self.N + 2, 1):
            if adj_array[i][j] > 0:
                adj_nodes += 1
                now_disperse += adj_array[i][j]
        if adj_nodes == 0:
            return 0
        else:
            return now_disperse / (adj_nodes * my_txr)

    def cal_used_energy_to_move(self, action):
        x = abs(action[0])
        y = abs(action[1])
        z = abs(action[2])
        energy_move = 11.2*(0.5*(x + y) + z)
        return energy_move

    def cal_used_energy_to_keep_txr(self, my_txr):
        energy_txr = math.pow(my_txr,2)
        return energy_txr

    def cal_reward(self, throughput, foot_of_perpendicular, dispersed, energy_move, energy_txr):
        u = 8  # constant that guarantees the reward to be non-negative
        foot_coeffi = 1/2
        reward = u + (foot_coeffi*throughput) + (foot_of_perpendicular) + (dispersed) - (energy_move/22) - (energy_txr/2)
        return reward

class My_DQN(gym.Env):
    metadata = {
        "render.modes": ["human"]
    }
    # ~number of relay node
    N = 2
    # ~transmission radius max
    R_MAX = 3
    # location x,y,z
    MIN_LOC = 0
    MAX_LOC = 4

    MIN_HEIGHT = 1
    MAX_HEIGHT = 4

    source = np.array((MIN_LOC, MIN_LOC, MIN_LOC, 2))
    dest = np.array((MAX_LOC, MAX_LOC, MAX_LOC, 0))
    agent2 = np.array((2,3,3,3))

    def __init__(self):
        low_range = (self.MIN_LOC, self.MIN_LOC, self.MIN_LOC, 0)
        high_range = (self.MAX_LOC, self.MAX_LOC, self.MAX_LOC, self.R_MAX)

        Low = np.array(low_range * (self.N + 2))
        High = np.array(high_range * (self.N + 2))

        self.observation_space = gym.spaces.Box(low=Low, high=High, dtype=int)
        self.action_space = gym.spaces.Discrete(81)

    def reset(self):
        self.count = 0
        '''if 21600 < epi:
            x = random.randint(self.MIN_LOC, self.MAX_LOC)
            y = random.randint(self.MIN_LOC, self.MAX_LOC)
            z = random.randint(self.MIN_HEIGHT, self.MAX_HEIGHT)
            r = random.randint(0, self.R_MAX)
        elif 5700 < epi:
            x = random.randint(self.MIN_LOC, self.MAX_LOC-1)
            y = random.randint(self.MIN_LOC, self.MAX_LOC-1)
            z = random.randint(self.MIN_HEIGHT, self.MAX_HEIGHT-1)
            r = random.randint(0, self.R_MAX)
        else :
            x = random.randint(self.MIN_LOC, self.MAX_LOC-2)
            y = random.randint(self.MIN_LOC, self.MAX_LOC-2)
            z = random.randint(self.MIN_HEIGHT, self.MAX_HEIGHT-2)
            r = random.randint(0, self.R_MAX)'''
        relay_node = []
        x = np.random.randint(self.MIN_LOC, self.MAX_LOC+1)
        y = np.random.randint(self.MIN_LOC, self.MAX_LOC+1)
        z = np.random.randint(self.MIN_HEIGHT, self.MAX_HEIGHT+1)
        r = np.random.randint(0, self.R_MAX+1)
        relay_node.append(x)
        relay_node.append(y)
        relay_node.append(z)
        relay_node.append(r)
        state_set = np.zeros((self.N + 2, 4), dtype=int)
        for i in range(4):
            state_set[0][i] = copy.deepcopy(relay_node[i])  # 릴레이노드
            state_set[1][i] = copy.deepcopy(self.agent2[i])
            state_set[2][i] = copy.deepcopy(self.source[i])
            state_set[3][i] = copy.deepcopy(self.dest[i])
        #state_set[0][2] = self.MIN_HEIGHT
        #state_set[0][3] = 2
        state_set = state_set.flatten()

        self.state = state_set
        self.last_set = np.zeros(9)
        self.reward = 0
        self.done = False
        self.info = {}

        return self.state

    def translate_action(self, action):
        array = np.zeros(4, dtype=int)
        for i in range(4):
            array[i] = action % 3
            action = action / 3
        for i in range(4):
            array[i] = array[i]-1
        return array

    def step(self, action):

        # print('my_env action : ', action)
        assert self.action_space.contains(action)
        self.count += 1
        real_action = self.translate_action(action)

        self.next_state = copy.deepcopy(self.state)
        # action을 모두 수행
        for i in range(4):
            self.next_state[i] += real_action[i]

        # x,y,z좌표 이동범위, txr 가능범위 넘었나 확인
            if (self.next_state[0 + 0] < self.MIN_LOC) or (self.MAX_LOC < self.next_state[0+0]):  # x좌표 이동범위 넘었나 확인
                self.next_state[0 + 0] -= real_action[0]
                real_action[0] = 0
            if (self.next_state[0 + 1] < self.MIN_LOC) or (self.MAX_LOC < self.next_state[0+1]):  # y좌표 이동범위 넘었나 확인
                self.next_state[0 + 1] -= real_action[1]
                real_action[1] = 0
            if (self.next_state[0 + 2] < self.MIN_HEIGHT) or (self.MAX_HEIGHT < self.next_state[0 + 2]):  # z좌표 이동범위 넘었나 확인
                self.next_state[0 + 2] -= real_action[2]
                real_action[2] = 0
            if (self.next_state[0 + 3] < 0) or ( self.R_MAX < self.next_state[0 + 3]):  # txr 가능범위 넘었나 확인
                self.next_state[0 + 3] -= real_action[3]
                real_action[3] = 0

        self.last_set[5] = real_action[0]
        self.last_set[6] = real_action[1]
        self.last_set[7] = real_action[2]
        self.last_set[8] = real_action[3]

        state_position_array = np.reshape(self.state, (self.N + 2, 4))
        next_position_array = np.reshape(self.next_state, (self.N + 2, 4))

        env = reward_set(self.N)
        adj_arr = env.cal_adjacency(next_position_array)
        self.throughput = env.cal_throughput(adj_arr)
        foot = env.cal_foot(next_position_array, self.source, self.dest, 0)
        dispersed = env.cal_dispersed(0, next_position_array[0][3], adj_arr)
        e_move = env.cal_used_energy_to_move(real_action)
        e_txr = env.cal_used_energy_to_keep_txr(next_position_array[0][3])

        self.last_set[0] = self.throughput
        self.last_set[1] = foot
        self.last_set[2] = dispersed
        self.last_set[3] = e_move
        self.last_set[4] = e_txr

        self.reward = env.cal_reward(self.throughput, foot, dispersed
                                     , e_move, e_txr)
        try:
            assert self.observation_space.contains(self.next_state)
        except AssertionError:
            print("INVALID STATE", self.next_state)

        self.state = self.next_state

        return [self.next_state, self.reward, self.done, self.last_set]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass
