from itertools import product

import gym
import numpy as np
from gym.utils import seeding
import random
import math
import networkx as nx
from matplotlib import pyplot as plt


class reward_set:
    def __init__(self, N):
        self.N = N

    #S-D까지 연결되는지 확인하기위해서 인접행렬 만들기
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
        #print('adj\n',adj_array)
        return adj_array

    # 인접그래프 그리기.
    # S-D까지의 경로가 있나 확인 -> 홉의 개수(has_path최단거리)찾기 -> throughput 구하기
    def cal_throughput(self, adj_array):
        graph = nx.Graph()
        for i in range(0, self.N + 2, 1):
            graph.add_node(i)
        for i in range(0, self.N + 2, 1):
            for j in range(0, self.N + 2, 1):
                if adj_array[i][j] > 0:
                    graph.add_edge(i, j)

        if nx.has_path(graph, self.N, self.N+1) :
            path_hop = self.N+1
        else:
            path_hop = np.inf

        # print("path_hop : ",path_hop)
        if path_hop != np.inf:
            throughput = 20 / path_hop
        else:
            throughput = 0

        return throughput

    def cal_dispersed(self, i, my_txr, adj_array):
        adj_nodes = 0
        now_disperse = 0
        for j in range(0, self.N + 2, 1):
            if adj_array[i][j] > 0:
                adj_nodes += 1
                now_disperse += adj_array[i][j]
        # print('@@ : ',my_txr[i][3])
        if adj_nodes == 0:
            return 0
        else:
            return now_disperse / (adj_nodes * my_txr)

    def cal_h(self, x, y, z, source, destination):
        kx = destination[0] - source[0]
        ky = destination[1] - source[1]
        kz = destination[2] - source[2]
        constant = (((kx * x) + (ky * y) + (kz * z)) / (math.pow(kx, 2) + math.pow(ky, 2) + math.pow(kz, 2)))
        h = math.sqrt(math.pow(constant * kx - x, 2) + math.pow(constant * ky - y, 2) + math.pow(constant * kz - z, 2))
        # print("h",h)
        return h

    # (시간t일때의 수선의 발 - 시간t+1일때 수선의 발)길이 구하기
    def cal_foot_of_perpendicular(self, state_array, next_state_array, source, destination, i):

        foot_of_perpendicular = self.cal_h(state_array[i][0], state_array[i][1], state_array[i][2], source, destination) \
                            - self.cal_h(next_state_array[i][0], next_state_array[i][1], next_state_array[i][2], source, destination) \
                            - state_array[0][3] + next_state_array[0][3]
        return foot_of_perpendicular

    def cal_used_energy_to_move(self, action):
        energy_move = math.sqrt((math.pow(action[0], 2) + math.pow(action[1], 2) + math.pow(action[2], 2)))
        return energy_move

    def cal_used_energy_to_keep_txr(self, my_txr):
        energy_txr = my_txr
        return energy_txr

    def cal_reward(self, throughput, dispersed, foot_of_perpendicular, energy_move, energy_txr):
        u = 5  # constant that guarantees the reward to be non-negative
        reward = 5 + (5*throughput) + dispersed + foot_of_perpendicular - energy_move - (energy_txr * (2 / 5))
        return reward


class My_DQN2(gym.Env):

    metadata = {
        "render.modes": ["human"]
    }

    MAX_STEPS = 50
    # ~number of relay node
    N = 2
    # ~transmission radius max
    R_MAX = 3
    # location x,y,z
    MIN_LOC = 0
    MAX_LOC = 4

    MAX_HEIGHT = 3
    MIN_HEIGHT = 0

    source = np.array((MIN_LOC, MIN_LOC, MIN_HEIGHT, R_MAX))
    dest = np.array((MAX_LOC, MAX_LOC, MAX_LOC, 0))
    agent2 = np.array((3, 3, 3, 3))

    def __init__(self):
        """low_range = (MIN_LOC, MIN_LOC, MIN_LOC, 0)
        high_range = (MAX_LOC, MAX_LOC, MAX_LOC, R_MAX)

        Low = np.array(low_range * (N + 2))
        High = np.array(high_range * (N + 2))

        d = [-1, 0, 1]
        all_state = []
        for i in range(N + 2):
            all_state.append(d)
        self.state_space = list(product(*all_state))
        num_state_space = len(self.state_space) #81"""

        self.observation_space = gym.spaces.Box(low=np.array([self.MIN_LOC, self.MIN_LOC, self.MIN_LOC, 0]),
                                                high=np.array([self.MAX_LOC, self.MAX_LOC, self.MAX_LOC, self.R_MAX]),
                                                dtype=int)

        #self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Discrete(3), gym.spaces.Discrete(3), gym.spaces.Discrete(3)))
        #self.action_space = gym.spaces.MultiDiscrete([3,3,3,3])
        self.action_space = gym.spaces.Discrete(81)

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.count = 0

        self.state = self.dest.copy()
        self.state[2] = self.MAX_HEIGHT
        self.last_set = np.zeros(9)
        self.reward = 0
        self.done = False
        self.info = {}

        return self.state

    def translate_action(self,action):
        array = np.zeros(4, dtype=int)
        for i in range(4):
            array[i] = action % 3
            action = action/3
        for i in range(4):
            array[i] = array[i]
        return array

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : Discrete

        Returns
        -------
        observation, reward, done, info : tuple
            observation (object) :
                an environment-specific object representing your observation of
                the environment.

            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.

            done (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)

            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """

        if self.done:
            # code should never reach this point
            print("EPISODE DONE!!!")

        elif self.count == self.MAX_STEPS:
            self.done = True

        else:

            # print('my_env action : ', action)
            assert self.action_space.contains(action)
            self.count += 1
            real_action = self.translate_action(action)
            #print(real_action)
            self.next_state = self.state
            # action을 모두 수행
            for i in range(4):
                self.next_state[i] += (real_action[i] - 1)

            # x,y,z좌표 이동범위, txr 가능범위 넘었나 확인
            for i in range(0, 2, 1):  # x,y좌표 이동범위 넘었나 확인
                if (self.next_state[i] > self.MAX_LOC) or (self.next_state[i] < self.MIN_LOC):
                    self.next_state[i] -= (real_action[i] - 1)
            if (self.next_state[2] > self.MAX_HEIGHT) or (self.next_state[2] < self.MIN_HEIGHT):  # z좌표 이동범위 넘었나 확인
                self.next_state[2] -= (real_action[2] - 1)
            if (self.next_state[3] > self.R_MAX) or (self.next_state[3] < 0):  # txr 가능범위 넘었나 확인
                self.next_state[3] -= (real_action[3] - 1)

            state_position_array = np.zeros((self.N + 2, 4))
            state_position_array[0] = self.state
            state_position_array[1] = self.agent2.copy()
            state_position_array[2] = self.source.copy()
            state_position_array[3] = self.dest.copy()

            next_position_array = np.zeros((self.N + 2, 4))
            next_position_array[0] = self.next_state
            next_position_array[1] = self.agent2.copy()
            next_position_array[2] = self.source.copy()
            next_position_array[3] = self.dest.copy()

            env = reward_set(self.N)
            adj_arr = env.cal_adjacency(next_position_array)
            self.throughput = env.cal_throughput(adj_arr)
            dispersed = env.cal_dispersed(0, next_position_array[0][3], adj_arr)
            foot = env.cal_foot_of_perpendicular(state_position_array, next_position_array, self.source, self.dest, 0)
            e_move = env.cal_used_energy_to_move(real_action)
            e_txr = env.cal_used_energy_to_keep_txr(next_position_array[0][3])
            # print("%6.3f %6.3f %6.3f %6.3f %3d" %(throughput, dispersed, foot, e_move, e_txr))
            self.last_set[0] = self.throughput
            self.last_set[1] = foot
            self.last_set[2] = dispersed
            self.last_set[3] = e_move
            self.last_set[4] = e_txr
            self.last_set[5] = real_action[0]-1
            self.last_set[6] = real_action[1]-1
            self.last_set[7] = real_action[2]-1
            self.last_set[8] = real_action[3]-1

            self.reward = env.cal_reward(self.throughput, dispersed, foot, e_move, e_txr)

        try:
            # assert self.observation_space.contains(self.state)
            assert self.observation_space.contains(self.next_state)

        except AssertionError:
            print("INVALID STATE", self.next_state)
        return [self.next_state, self.reward, self.done, self.last_set]

    def render (self, state, mode="human"):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with
        """
        #s = "position: {:2d}  reward: {:2d}  info: {}"
        #print(s.format(self.state, self.reward, self.info))


    def seed (self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def close (self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass