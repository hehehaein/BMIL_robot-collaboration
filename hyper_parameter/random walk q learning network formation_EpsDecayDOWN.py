import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import time
import math

"""
basic setup
1. environment
- [S 0 1]
- [2 3 4]
- [5 6 D]
- 1x1 block for each agent
- 3x3 network blocck
2. state, action, reward
- state = {number of neighbors} = {0 ~ 8}
- action = {changing transmission range} = {-1, 0, 1}
- reward = {throughput improvement} + {panelty for energy consumption}
- transmission range = {0, 1, 2}
3. random walk
- moving distance ~ U(0, 0.01)
- moving angle ~ U(0, 360)
"""

max_node = 9  # basic setup : 전체 노드 수를 미리 선언해야 계산 용이


# 전체 environment
class NodeEnv:
    def __init__(self, txr_level, txr_interval):
        # envorionment matrix에 해당하는 변수 선언
        self.net_dim = 3
        self.one_dim = 1
        self.max_dim = self.net_dim * self.one_dim
        self.net_block = self.net_dim ** 2  # 정사각형 environment 가정
        self.num_node = self.net_block

        # default block coordination 선언
        self.block_coords = np.zeros((self.net_block, 2))
        i = 0
        for x in range(0, self.max_dim, self.one_dim):
            for y in range(0, self.max_dim, self.one_dim):
                self.block_coords[i, :] = [x, y]
                i += 1

        self.net_block_coords = self.block_coords.tolist()

        # agent location 지정
        self.node_loc = []
        for i in range(self.num_node):
            self.node_loc.append(self.net_block_coords[i])

        # source, destination 제외
        del self.node_loc[0]
        del self.node_loc[self.num_node - 2]

        # tx range, action 변수 선언
        self.txr_level = txr_level # 1
        self.txr_interval = txr_interval # 1
        self.max_txr = self.txr_level * self.txr_interval # max_txr=1
        self.action_space = np.arange(-self.max_txr, self.max_txr + 1, self.txr_interval) # (-1, 2, 1)
        self.num_action = len(self.action_space)

    @staticmethod
    def DistM(node_loc):  # distance matrix 계산
        dist_M = np.zeros([max_node, max_node])
        for i in range(max_node):
            for j in range(max_node):
                dist_M[i, j] = np.linalg.norm(np.array(node_loc[i]) - np.array(node_loc[j]))
        return dist_M

    @staticmethod  # adjacency matrix 계산
    def AdjM(dist_M, tx_range):
        state_adj_M = np.zeros([max_node, max_node])

        for i in range(max_node):
            state_adj_M[i] = ((dist_M[i]) <= tx_range[i])

        return state_adj_M

    @staticmethod
    def get_State(state_adj_M):  # node 별로 state 계산 (자기자신은 제외)
        sum_adj_M = np.zeros([max_node])
        sum_adj_M = state_adj_M.sum(axis=1)
        next_state = np.zeros([max_node])

        for i in range(max_node):
            next_state[i] = sum_adj_M[i] - 1

        return next_state

    @staticmethod
    def get_Throughput(adj_M, dist_M):  # network throughput 계산
        dest = max_node - 1
        source = 0

        net_G = nx.DiGraph() # 방향성그래프 만들기
        for i in range(max_node):
            net_G.add_node(i)
            for j in range(max_node):
                if adj_M[i, j] == 1:
                    # weight 적어주기
                    net_G.add_weighted_edges_from([(i, j, dist_M[i, j])])

        if nx.has_path(net_G, source, dest):
            #최단경로 찾아주기
            path_txr = nx.shortest_path_length(net_G, source, dest)

        else:
            path_txr = np.inf

        throughput = 10 / path_txr  # scaling factor로 source to destination 거리 나누기

        return throughput

    def takeStep(self, action, next_txr, last_throughput, node_loc):  # action 수행 결과 도출
        node_loc = node_loc.tolist()
        # 노드 movement 상황
        node_loc.insert(0, [0.0, 0.0])  # source location 추가
        node_loc.append([2.0, 2.0])  # destination location 추가

        next_txr = next_txr.tolist()
        next_txr.insert(0, 1.5)  # 리스트 내 index =0인 자리에 1.5 추가
        next_txr.append(0.0)  # 리스트 내 index 마지막 자리에 0 추가

        next_dist_M = self.DistM(node_loc) # 노드간의 거리 계산한 array
        next_adj_M = self.AdjM(next_dist_M, next_txr) # 인접한 노드들

        next_state = self.get_State(next_adj_M) # 인접한 노드가 몇개인지
        cur_throughput = self.get_Throughput(next_adj_M, next_dist_M)

        # get reward
        pos_coeff = 1
        balance_w = 0.45  # 0.45: throughput 보장하면서 action 최적화하는 최솟값 discount factor
        net_reward = np.zeros([max_node - 2])

        for i in range(max_node - 2): # reward 계산
            net_reward[i] = pos_coeff + balance_w * (cur_throughput - last_throughput) - (1 - balance_w) * action[i]

        node_loc.remove([0.0, 0.0])
        node_loc.remove([2.0, 2.0])
        node_loc = np.array(node_loc)

        return next_state, net_reward, cur_throughput

    # episode 시작할 때마다 environment reset
    def reset(self):
        now_txr = np.zeros([max_node - 2])
        current_state = np.zeros([max_node - 2])
        cur_loc = self.node_loc
        last_throughput = 0

        return current_state, now_txr, cur_loc, self.node_loc, last_throughput


class QlearningNode:
    def __init__(self, action_space, q_table):
        self.learningrate = 1e-3
        self.discount_factor = 0.9
        self.action_space = action_space
        self.num_action = len(self.action_space)

        self.q_table = q_table

        # 무빙 에이전트에 해당 하는 변수 선언
        self.node_net_size = 1  # block 밖으로 나가지 못하게 제한하기 위함

    def print_qtable(self):
        print(self.q_table)

    def learn(self, epoch, state, action, reward, next_state):
        state = int(state)
        action_space = self.action_space.tolist()
        action_idx = action_space.index(action)
        current_q = self.q_table[state, action_idx]
        next_q = reward + self.discount_factor * np.max(self.q_table[int(next_state), :])
        self.q_table[state, action_idx] += self.learningrate * (next_q - current_q)

    def getAction(self, cur_state, eps, cur_txr): # epslion greedy 사용
        if (eps > np.random.rand()):  # uniform random action selection
            action_idx = np.random.randint(0, 3)
            action = self.action_space[int(action_idx)]
            next_txr = cur_txr + action
        else:
            # 구해놓은 Q table을 사용
            state_action = self.q_table[int(cur_state), :]
            action_idx = np.argmax(state_action)
            action = self.action_space[int(action_idx)]
            next_txr = cur_txr + action

        # 정해진 tx range 범위 내에서만 action 고르도록...?
        while next_txr < 0 or next_txr >= 3:
            action_idx = np.random.randint(0, 3)
            action = self.action_space[int(action_idx)]
            next_txr = cur_txr + action

        return action, action_idx, next_txr

    # movement part
    def RandomWalk(self, original_node_loc, now_node_loc):
        random_direction = np.random.randint(0, 360)
        moving_distance = np.random.rand() * 0.01

        if random_direction == 360:
            next_node_loc = now_node_loc

        else:
            next_node_loc = [moving_distance * math.cos((random_direction)) + now_node_loc[0],
                             moving_distance * math.sin((random_direction)) + now_node_loc[1]]

        # 할당된 block 벗어나지 않도록 x, y 좌표 확인
        row = next_node_loc[0]
        col = next_node_loc[1]

        ori_row = original_node_loc[0]
        ori_col = original_node_loc[1]

        lim_row = self.node_net_size + ori_row
        lim_col = self.node_net_size + ori_col

        if (row >= ori_row) and (row <= lim_row) and (col >= ori_col) and (col <= lim_col):
            return next_node_loc

        else:  # block 벗어나면 다시 뽑기
            try:
                return self.RandomWalk(original_node_loc, now_node_loc)
            except RecursionError:
                return now_node_loc


if __name__ == '__main__':

    tic = time.time()

    txr_level = 1  # tx range 총 단계
    txr_interval = 1  # tx range 간격

    env = NodeEnv(txr_level, txr_interval)
    action_space = env.action_space

    episode = 0  # 1 episode = 100 step
    num_episodes = 200000
    num_step = 100

    # epsilon greedy method
    eps_end = 0.01
    eps_start = 1.0
    eps_decay = 1e-5
    eps_trace = []

    # trace method
    epi_reward = 0
    last_reward = 0
    reward = []
    epi_through = 0
    throughput_trace = []
    reward_trace = []
    txr_step = np.zeros([num_step, max_node - 2])  # 스텝별 tx range 모두 저장
    txr_node = np.zeros([num_episodes, max_node - 2])  # episode의 평균 tx range

    next_loc = np.zeros([max_node - 2, 2])  # 노드의 x, y축 좌표

    q_table = np.zeros([max_node - 2, max_node, len(action_space)])  # (총 agent 수, state, action space)

    node = []
    for i in range(max_node - 2):  # for all agents
        node.append(QlearningNode(action_space, q_table[i, :, :]))

    for epi in range(num_episodes):

        state, cur_txr, cur_node_loc, ori_node_loc, last_throughput = env.reset()
        cur_node_loc = np.array(cur_node_loc)
        ori_node_loc = np.array(ori_node_loc)

        # epsilon greedy : episode 진행에 따라 explore 비율 감소
        eps = max(eps_end, eps_start - (eps_decay * epi))
        eps_trace.append(eps)

        # trace method
        reward_trace.append(epi_reward / num_step)
        throughput_trace.append(epi_through / num_step)
        txr_node[epi, :] = np.round(txr_step.sum(axis=0) / num_step)
        txr_step = np.zeros([num_episodes, max_node - 2])
        epi_reward = 0
        epi_through = 0

        for step in range(num_step):
            # initializing
            action_idx = np.zeros([max_node - 2])
            next_txr = np.zeros([max_node - 2])
            action = np.zeros([max_node - 2])

            # epsilon greedy policy에 의해 action select
            for i in range(max_node - 2):
                action[i], action_idx[i], next_txr[i] = node[i].getAction(state[i], eps, cur_txr[i])

            next_state, reward, goodput = env.takeStep(action, next_txr, last_throughput, cur_node_loc)

            # source, destination은 state 개념 없으므로 삭제
            next_state = np.delete(next_state, max_node - 1)
            next_state = np.delete(next_state, 0)

            for i in range(max_node - 2):
                node[i].learn(epi, state[i], action[i], reward[i], next_state[i])
                next_loc[i] = node[i].RandomWalk(ori_node_loc[i], cur_node_loc[i])  # random walk model

            state = next_state
            cur_txr = next_txr
            last_throughput = goodput
            cur_node_loc = next_loc
            epi_reward += (sum(reward) / (max_node - 2))  # 모든 에이전트들의 평균 reward
            epi_through += last_throughput
            txr_step[step, :] = next_txr

    # print every result
    toc = time.time()
    print("processing time", toc - tic)

    print("\n")
    print("eps ", eps)

    for i in range(max_node - 2):
        print("%s node" % i)
        node[i].print_qtable()

    # 전체 에피소드에 대해 50개 단위로 나누어 평균 취해서 그래프 plot
    reward_trace_sum = []
    for i in range(0, num_episodes, 50):
        reward_trace_sum.append(sum(reward_trace[i:i + 50]) / 50)
    throughput_trace_sum = []
    for i in range(0, num_episodes, 50):
        throughput_trace_sum.append(sum(throughput_trace[i:i + 50]) / 50)
    reward_trace.pop(0)
    throughput_trace.pop(0)
    reward_trace_sum.pop(0)
    throughput_trace_sum.pop(0)

    plt.figure()
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.plot(reward_trace)

    plt.figure()
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.plot(reward_trace_sum)

    plt.figure()
    plt.xlabel('episode')
    plt.ylabel('epsilon')
    plt.title('epsilon')
    plt.plot(eps_trace)

    plt.figure()
    plt.xlabel('Episodes')
    plt.ylabel('Throughput')
    plt.plot(throughput_trace)

    plt.figure()
    plt.xlabel('Episodes')
    plt.ylabel('Throughput')
    plt.plot(throughput_trace_sum)

    plt.figure()
    plt.xlabel('agents')
    plt.ylabel('Episodes')
    a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t = np.vsplit(txr_node, 20)
    a_sum = a.mean(axis=0)
    b_sum = b.mean(axis=0)
    c_sum = c.mean(axis=0)
    d_sum = d.mean(axis=0)
    e_sum = e.mean(axis=0)
    f_sum = f.mean(axis=0)
    g_sum = g.mean(axis=0)
    h_sum = h.mean(axis=0)
    i_sum = i.mean(axis=0)
    j_sum = j.mean(axis=0)
    k_sum = k.mean(axis=0)
    l_sum = l.mean(axis=0)
    m_sum = m.mean(axis=0)
    n_sum = n.mean(axis=0)
    o_sum = o.mean(axis=0)
    p_sum = p.mean(axis=0)
    q_sum = q.mean(axis=0)
    r_sum = r.mean(axis=0)
    s_sum = s.mean(axis=0)
    t_sum = t.mean(axis=0)
    # print(a_sum)
    z_1 = np.vstack((a_sum, b_sum))
    z_2 = np.vstack((z_1, c_sum))
    z_3 = np.vstack((z_2, d_sum))
    z_4 = np.vstack((z_3, e_sum))
    z_5 = np.vstack((z_4, f_sum))
    z_6 = np.vstack((z_5, g_sum))
    z_7 = np.vstack((z_6, h_sum))
    z_8 = np.vstack((z_7, i_sum))
    z_9 = np.vstack((z_8, j_sum))
    z_10 = np.vstack((z_9, k_sum))
    z_11 = np.vstack((z_10, l_sum))
    z_12 = np.vstack((z_11, m_sum))
    z_13 = np.vstack((z_12, n_sum))
    z_14 = np.vstack((z_13, o_sum))
    z_15 = np.vstack((z_14, p_sum))
    z_16 = np.vstack((z_15, q_sum))
    z_17 = np.vstack((z_16, r_sum))
    z_18 = np.vstack((z_17, s_sum))
    z = np.vstack((z_18, t_sum))
    # print(z[:,20:41])
    # print(z)
    """x = np.arange(0, 23, 1)
    y = np.arange(0, 20, 0.5)
    plt.imshow(z, vmin=0, vmax = 3)
    plt.colorbar()
    plt.xlim([0,22])
    plt.ylim([0,19])
    plt.grid(True)
    plt.xticks(x)
    plt.yticks(y)
    #plt.title("3x3 random walk")
    plt.show()"""
    x = np.arange(0, 7, 1) #23->14
    y = np.arange(0, 20, 0.5)
    plt.imshow(z[:, 0:20], vmin=0, vmax=2) #vmax 8->2
    plt.colorbar()
    plt.xlim([0, 7]) #19->14
    plt.ylim([0, 20]) #19->20
    plt.grid(True)
    plt.xticks(x)
    plt.yticks(y)
    plt.show()