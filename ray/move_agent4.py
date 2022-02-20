
import numpy as np
import networkx as nx
import time
import math
from itertools import product
from matplotlib import pyplot as plt
from scipy import stats


# parameter
eps_end = 0.01
eps_start = 1.0
eps_decay = 1e-5 * 5

num_episode = 40000
num_step = 150

# environment
num_agent = 4
num_sd = 2
num_node = num_agent + num_sd
net_dim = 5
env_net = np.zeros([net_dim, net_dim])
source_loc = [0, 0]
dest_loc = [5, 5]
init_loc = [[5, 5], [5, 5], [5, 5], [5, 5]]
max_txr = 2 # node 자체의 최대 tx radius
max_moving = 2 # 1 step에서 최대 이동 거리
interval_moving =1  # 1 step에서의 이동 거리 변화량

# state
d = [np.inf, 0, 1, 2]
all_state = [d, d, d, d, d]
all_state_space = list(product(*all_state))  # 전체 state 조합
array_of_tuples = map(tuple, all_state_space)
state_space = tuple(array_of_tuples)
num_state = len(state_space)


# action
mov_dist_space = np.arange(1, max_moving + interval_moving, interval_moving)  # 이동 거리
mov_angle_space = np.arange(0, 4, 1)  # 이동 각도
all_action = [mov_dist_space, mov_angle_space]
action_space_moving = list(product(*all_action))  # 전체 action 조합
non_moving = [(0,0)]
action_space = non_moving + action_space_moving
num_action = len(action_space)

# agent
class QlearningNode:
    def __init__(self, action_space, q_table):
        self.learningrate = 1e-2
        self.discount_factor = 0.8
        self.action_space = action_space
        self.num_action = len(self.action_space)
        self.q_table = q_table
        self.visit_state = []

    def return_qtable(self):
        return self.q_table

    def return_visitstate(self):
        return self.visit_state

    def learn(self, state_idx, action_idx, reward, next_state_idx):  # Q table update
        state = int(state_idx)
        action_idx = int(action_idx)
        current_q = self.q_table[state, action_idx]
        next_q = reward + self.discount_factor * np.max(self.q_table[int(next_state_idx), :])
        self.q_table[state, action_idx] += self.learningrate * (next_q - current_q)

        if np.isnan(self.q_table[state, action_idx]):
            print("state", state)
            print("action idx", action_idx)
            print("next q", next_q)
            print("cur q", current_q)
        self.visit_state.append(state)

    def getAction(self, eps, cur_state_idx, cur_loc, cur_angle):
        if (eps > np.random.rand()):  # uniform random action selection + epsilon-greedy
            action_idx = np.random.randint(0, num_action)
            action = self.action_space[int(action_idx)]
            dist_action = action[0]
            angle_action = action[1]
        else:
            state_action = self.q_table[int(cur_state_idx), :]  # argmax action selection
            action_idx = np.argmax(state_action)
            action = self.action_space[int(action_idx)]
            dist_action = action[0]
            angle_action = action[1]

        moving_distance = dist_action
        moving_angle = angle_action * 90
        # select된 action에 따른 다음 위치 계산
        next_loc = [moving_distance * math.cos((moving_angle)) + cur_loc[0],
                    moving_distance * math.sin((moving_angle)) + cur_loc[1]]

        # 정해진 network 벗어나지 않도록 action 재선택
        while next_loc[0] < 0 or next_loc[0] > net_dim or next_loc[1] < 0 or next_loc[1] > net_dim:
            action_idx = np.random.randint(0, num_action)
            action = self.action_space[int(action_idx)]
            dist_action = action[0]
            angle_action = action[1]
            moving_distance = dist_action
            moving_angle = angle_action * 90
            next_loc = [moving_distance * math.cos((moving_angle)) + cur_loc[0],
                        moving_distance * math.sin((moving_angle)) + cur_loc[1]]

        if moving_angle >= 360:
            moving_angle -= 360

        return action_idx, dist_action, angle_action, next_loc, moving_angle


# function
def DistM(node_loc):  # distance matrix 계산
    dist_M = np.zeros([num_node, num_node])
    for i in range(num_node):
        for j in range(num_node):
            dist_M[i, j] = np.rint(np.linalg.norm(np.array(node_loc[i]) - np.array(node_loc[j])))
    return dist_M

def get_NextState(dist_M, tx_range):
    state_dist_M = np.zeros([num_node, num_node])
    state_angle_M = np.zeros([num_node, num_node])
    for i in range(num_node):
        for j in range(num_node):
            if dist_M[i, j] > tx_range[i]:
                state_dist_M[i, j] = np.inf
            else:
                state_dist_M[i, j] = np.round(dist_M[i, j]/1)
    return state_dist_M


def AdjM(dist_M, tx_range):  # transmission radius와 distance matrix로 노드 간 연결성 계산
    state_adj_M = np.zeros([num_node, num_node])
    for i in range(num_node):
        state_adj_M[i] = ((dist_M[i]) <= tx_range[i])
    return state_adj_M

def get_Neighbor(state_adj_M):  # neighbor 수 계산 (tranmission radius 안에 존재하는가?)
    sum_adj_M = np.zeros(num_node)
    sum_adj_M = state_adj_M.sum(axis=1)
    next_neighbor = np.zeros(num_agent)
    for i in range(num_agent):
        next_neighbor[i] = sum_adj_M[i + 1] - 1
    return next_neighbor

def get_Shortest(adj_M, dist_M):  # shortest path in environment
    dest = num_node - 1
    source = 0
    net_G = nx.Graph()
    for i in range(num_node):
        net_G.add_node(i)
        for j in range(num_node):
            if adj_M[i, j] == 1:
                net_G.add_edge(i, j, distance=dist_M[i, j])

    if nx.has_path(net_G, source, dest):
        min_hop = nx.shortest_path_length(net_G, source, dest)
        min_path = nx.dijkstra_path_length(net_G, source, dest)
    else:
        min_hop = np.inf
        min_path = np.inf
    total_min_hop = min_hop
    total_min_path = min_path
    return total_min_hop, total_min_path

def get_SumDistance(dist_M, adj_M):
    neigh_dist_sum = np.zeros(num_node)
    for i in range(num_node):
        for j in range(num_node):
            if adj_M[i, j] == 1:
                neigh_dist_sum[i] += dist_M[i, j]
    dist_sum = np.delete(neigh_dist_sum, (num_node - 1), axis=0)
    dist_sum = np.delete(dist_sum, (0), axis=0)
    return dist_sum

def get_Diff(adj_M, dist_M, agent):
    agent_adj_M = np.delete(adj_M, agent, 0)
    agent_adj_M = np.delete(adj_M, agent, 1)
    agent_dist_M = np.delete(dist_M, agent, 0)
    agent_dist_M = np.delete(dist_M, agent, 1)
    dest = num_node - 2
    source = 0
    net_G = nx.Graph()
    for i in range(num_node-1):
        net_G.add_node(i)
        for j in range(num_node-1):
            if agent_adj_M[i, j] == 1:
                net_G.add_weighted_edges_from([(i, j, agent_dist_M[i, j])])
    if nx.has_path(net_G, source, dest):
        min_hop = nx.shortest_path_length(net_G, source, dest)
        min_path = nx.dijkstra_path_length(net_G, source, dest)
    else:
        min_hop = np.inf
        min_path = np.inf
    return min_hop, min_path

def takeStep(epi, dist_action, angle_action, next_txr, node_loc, next_angle):  # action 수행 결과 도출
    node_loc = node_loc.tolist()
    node_loc.insert(0, source_loc)  # source location 추가
    node_loc.append(dest_loc)  # destination location 추가

    next_txr = next_txr.tolist()
    next_txr.insert(0, 1)  # source transmission radius 추가
    next_txr.append(0)

    next_angle = next_angle.tolist()
    next_angle.insert(0, 0)
    next_angle.append(0)

    next_dist_M = DistM(node_loc)
    state_dist_M= get_NextState(next_dist_M, next_txr)

    next_adj_M = AdjM(next_dist_M, next_txr)
    neigh_dist_sum = get_SumDistance(next_dist_M, next_adj_M)
    next_neighbor = get_Neighbor(next_adj_M)
    shortest_hop, shortest_path = get_Shortest(next_adj_M, next_dist_M)

    next_txr.remove(1)
    next_txr.remove(0)
    next_txr = np.array(next_txr)

    next_angle.remove(0)
    next_angle.remove(0)
    next_angle = np.array(next_angle)

    state_dist = state_dist_M.tolist()
    del state_dist[num_node - 1]
    del state_dist[0]

    for i in range(num_agent):
        del state_dist[i][i]

    next_state_list = []
    for i in range(num_agent):
        next_state = state_dist[i]
        next_state_list.append(next_state)

    next_state_array = np.array(next_state_list)
    array_of_tuples = map(tuple, next_state_array)
    next_state = tuple(array_of_tuples)
    next_state_idx = np.zeros(num_agent)
    for i in range(num_agent):
        next_state_idx[i] = state_space.index(next_state[i])

    # get reward
    reward = np.zeros(num_agent)
    diff_reward = np.zeros(num_agent)
    for i in range(num_agent):
        diff_hop, diff_path = get_Diff(next_adj_M, next_dist_M, i)
        diff_reward[i] = shortest_path - diff_path
        if np.isnan(diff_reward[i]):
            diff_reward[i] = 0
        elif np.isinf(diff_reward[i]):
            diff_reward[i] = net_dim +3

    throughput = 10 / shortest_hop
    moving_weight = epi * 1e-5

    dist_cost = np.zeros(num_agent)
    angle_cost = np.zeros(num_agent)
    if epi > (num_episode / 3) and epi < (num_episode - (num_episode / 3)):
        for i in range(num_agent):
            if dist_action[i] == 2:
                dist_cost[i] = 1
            if angle_action[i] == 3:
                angle_cost[i] = 1

    if epi > (num_episode - (num_episode / 3)):
        for i in range(num_agent):
            dist_cost[i] = dist_action[i]
            angle_cost[i] = angle_action[i]

    dense_reward = np.zeros(num_agent)
    for i in range(num_agent):
        if next_neighbor[i] == 0:
            dense_reward[i] = 0
        elif neigh_dist_sum[i] == 0 or next_txr[i]==0:
            dense_reward[i] = 0
        else:
            dense_reward[i] = neigh_dist_sum[i] / (next_neighbor[i] * next_txr[i])

        reward[i] = - ((dist_cost[i] + angle_cost[i] + next_txr[i]) * moving_weight) + (20 * throughput) + ((15- moving_weight) * (dense_reward[i]))

    node_loc.remove(source_loc)
    node_loc.remove(dest_loc)
    node_loc = np.array(node_loc)

    return next_state_idx, reward, throughput, diff_reward, dense_reward


def reset():
    all_loc = init_loc
    all_loc.insert(0, source_loc)  # source location 추가
    all_loc.append(dest_loc)  # destination location 추가

    now_txr = np.zeros(num_agent)
    for i in range(num_agent):
        now_txr[i] = max_txr  # max tx radius

    now_angle = np.zeros(num_agent)

    now_txr = now_txr.tolist()
    now_txr.insert(0, 1)  # 리스트 내 index = 0인 자리에 1 추가 (source tx radius = 1)
    now_txr.append(0)  # 리스트 내 index 마지막 자리에 0 추가

    now_angle = now_angle.tolist()
    now_angle.insert(0, 0)
    now_angle.append(0)

    now_dist_M = DistM(all_loc)
    now_adj_M = AdjM(now_dist_M, now_txr)
    state_dist_M = get_NextState(now_dist_M, now_txr)

    state_dist = state_dist_M.tolist()
    for i in range (num_node):
        del state_dist[i][i]

    now_state_list = []
    for i in range(num_agent):
        state = state_dist[i]
        now_state_list.append(state)

    now_state_array = np.array(now_state_list)
    array_of_tuples = map(tuple, now_state_array)
    now_state = tuple(array_of_tuples)
    cur_obs_idx = np.zeros(num_agent)
    for i in range(num_agent):
        try:
            cur_obs_idx[i] = state_space.index(now_state[i])
        except ValueError:
            print("value error", now_state)

    now_txr.remove(1)
    now_txr.remove(0)
    now_txr = np.array(now_txr)

    now_angle.remove(0)
    now_angle.remove(0)
    now_angle = np.array(now_angle)

    all_loc.remove(source_loc)
    all_loc.remove(dest_loc)
    cur_loc = np.array(all_loc)

    return cur_obs_idx, now_txr, cur_loc, now_angle


# main
cur_txr = np.zeros(num_agent)
next_loc = np.zeros([num_agent, 2])  # 노드의 x, y축 좌표
cur_angle = np.zeros(num_agent)

# trace method
epi_reward1 = 0
last_reward1 = 0
reward1_trace = []
epi_reward2 = 0
last_reward2 = 0
reward2_trace = []
epi_reward3 = 0
last_reward3 = 0
reward3_trace = []
epi_reward4 = 0
last_reward4 = 0
reward4_trace = []
epi_reward5 = 0
last_reward5 = 0
reward5_trace = []

dist_step = np.zeros([num_step, num_agent])  # 스텝별 tx range 모두 저장
dist_node = np.zeros([num_episode, num_agent])  # episode의 평균 tx range
throughput_trace = []
epi_through = 0

dense1_trace = []
epi_dense1 = 0
dense2_trace = []
epi_dense2 = 0
dense3_trace = []
epi_dense3 = 0
dense4_trace = []
epi_dense4 = 0
dense5_trace = []
epi_dense5 = 0

dist_trace = []
epi_dist = 0

angle_trace = []
epi_angle = 0

loc_step = np.zeros([num_step, num_agent, 2])
loc_node_all = np.zeros([num_step*num_episode, num_agent, 2])
loc_node = np.zeros([num_episode, num_agent, 2])

q_table = np.zeros([num_agent, num_state, num_action])
agent = []
for i in range(num_agent):  # for all agents
    agent.append(QlearningNode(action_space, q_table[i, :, :]))

tic = time.time()

for epi in range(num_episode):

    print("episode", epi, ": throughput", epi_through, ": reward", epi_reward1, ": loc0", np.round(loc_step[:,0,:].sum(axis=0) / num_step, 2))

    # epsilon greedy : episode 진행에 따라 explore 비율 감소
    eps = max(eps_end, eps_start - (eps_decay * epi))
    cur_obs_idx, cur_txr, cur_loc, cur_angle = reset()

    reward1_trace.append(epi_reward1 / num_step)
    epi_reward1 = 0
    reward2_trace.append(epi_reward2 / num_step)
    epi_reward2 = 0
    reward3_trace.append(epi_reward3 / num_step)
    epi_reward3 = 0
    reward4_trace.append(epi_reward4 / num_step)
    epi_reward4 = 0
    reward5_trace.append(epi_reward5 / num_step)
    epi_reward5 = 0

    dense1_trace.append(epi_dense1 / num_step)
    epi_dense1 = 0
    dense2_trace.append(epi_dense2 / num_step)
    epi_dense2 = 0
    dense3_trace.append(epi_dense3 / num_step)
    epi_dense3 = 0
    dense4_trace.append(epi_dense4 / num_step)
    epi_dense4 = 0
    dense5_trace.append(epi_dense5 / num_step)
    epi_dense5 = 0

    throughput_trace.append(epi_through / num_step)
    epi_through = 0

    dist_trace.append(epi_dist / num_step)
    epi_dist = 0

    angle_trace.append(epi_angle / num_step)
    epi_angle = 0

    #loc_node[epi, :, :] = np.round(loc_step.sum(axis=0) / num_step, 2)
    loc_node[epi, :, :] = stats.mode(loc_step)[0]
    #loc_node[epi, :, :] = loc_step[num_step-1]
    loc_step = np.zeros([num_step, num_agent, 2])

    last_throughput = 0

    for step in range(num_step):

        action_idx = np.zeros(num_agent)
        next_txr = np.zeros(num_agent)
        for i in range(num_agent):
            next_txr[i] = max_txr

        action = np.zeros(num_agent)
        txr_action = np.zeros(num_agent)
        dist_action = np.zeros(num_agent)
        angle_action = np.zeros(num_agent)
        next_angle = np.zeros(num_agent)

        # epsilon greedy policy에 의해 action select
        for i in range(num_agent):
            action_idx[i], dist_action[i], angle_action[i], next_loc[i], next_angle[i] = agent[i].getAction(eps, cur_obs_idx[i], cur_loc[i], cur_angle[i])

        next_obs_idx, reward, throughput, diff_reward, dense_reward = takeStep(epi, dist_action, angle_action, next_txr, next_loc, next_angle)

        for i in range(num_agent):
            agent[i].learn(cur_obs_idx[i], action_idx[i], reward[i], next_obs_idx[i])

        #state = next_state
        cur_obs_idx = next_obs_idx
        cur_loc = next_loc
        cur_angle = next_angle
        cur_txr = next_txr

        epi_reward1 += reward[0]
        epi_reward2 += reward[1]
        epi_reward3 += reward[2]
        epi_reward4 += reward[3]
        #epi_reward5 += reward[4]

        epi_dense1 += dense_reward[0]
        epi_dense2 += dense_reward[1]
        epi_dense3 += dense_reward[2]
        epi_dense4 += dense_reward[3]
        #epi_dense5 += dense_reward[4]

        last_throughput = throughput

        epi_through += throughput
        epi_dist += dist_action
        epi_angle += angle_action
        loc_step[step, :, :] = cur_loc
        loc_node_all[(step*epi)+step, :, :] = cur_loc


# print every result
toc = time.time()
print("processing time", toc - tic)
#np.savetxt('txr=2+coeff2.txt', throughput_location, fmt="%s")

reward1_trace_sum = []
for i in range(0, num_episode, 50):
    reward1_trace_sum.append(sum(reward1_trace[i:i + 50]) / 50)

reward2_trace_sum = []
for i in range(0, num_episode, 50):
    reward2_trace_sum.append(sum(reward2_trace[i:i + 50]) / 50)

reward3_trace_sum = []
for i in range(0, num_episode, 50):
    reward3_trace_sum.append(sum(reward3_trace[i:i + 50]) / 50)

reward4_trace_sum = []
for i in range(0, num_episode, 50):
    reward4_trace_sum.append(sum(reward4_trace[i:i + 50]) / 50)

#reward5_trace_sum = []
#for i in range(0, num_episode, 50):
#    reward5_trace_sum.append(sum(reward5_trace[i:i + 50]) / 50)

dense1_trace_sum = []
for i in range(0, num_episode, 50):
    dense1_trace_sum.append(sum(dense1_trace[i:i + 50]) / 50)

dense2_trace_sum = []
for i in range(0, num_episode, 50):
    dense2_trace_sum.append(sum(dense2_trace[i:i + 50]) / 50)

dense3_trace_sum = []
for i in range(0, num_episode, 50):
    dense3_trace_sum.append(sum(dense3_trace[i:i + 50]) / 50)

dense4_trace_sum = []
for i in range(0, num_episode, 50):
    dense4_trace_sum.append(sum(dense4_trace[i:i + 50]) / 50)

# dense5_trace_sum = []
# for i in range(0, num_episode, 50):
#     dense5_trace_sum.append(sum(dense5_trace[i:i + 50]) / 50)

throughput_trace_sum = []
for i in range(0, num_episode, 50):
    throughput_trace_sum.append(sum(throughput_trace[i:i + 50]) / 50)

angle_trace_sum = []
for i in range(0, num_episode, 50):
    angle_trace_sum.append(sum(angle_trace[i:i + 50]) / 50)

throughput_trace_sum = []
for i in range(0, num_episode, 50):
    throughput_trace_sum.append(sum(throughput_trace[i:i + 50]) / 50)

dist_trace_sum = []
for i in range(0, num_episode, 50):
    dist_trace_sum.append(sum(dist_trace[i:i + 50]) / 50)

# np.savetxt('agent4_loc0.txt', loc_step[:,0,:])
# np.savetxt('agent4_loc1.txt', loc_step[:,1,:])
# np.savetxt('agent4_loc2.txt', loc_step[:,2,:])
# np.savetxt('agent4_loc3.txt', loc_step[:,3,:])
#
# np.savetxt('agent4_locnode0.txt', loc_node[:,0,:])
# np.savetxt('agent4_locnode1.txt', loc_node[:,1,:])
# np.savetxt('agent4_locnode2.txt', loc_node[:,2,:])
# np.savetxt('agent4_locnode3.txt', loc_node[:,3,:])

plt.figure()
plt.title("average reward")
plt.plot(reward1_trace_sum, label='agent 0')
plt.plot(reward2_trace_sum, label='agent 1')
plt.plot(reward3_trace_sum, label='agent 2')
plt.plot(reward4_trace_sum, label='agent 3')
# plt.plot(reward5_trace_sum, label='agent 4')
plt.legend()
plt.show()

plt.figure()
plt.title("average throughput")
plt.plot(throughput_trace_sum)
plt.show()

plt.figure()
plt.title("throughput")
plt.plot(throughput_trace)
plt.show()

plt.figure()
plt.title("distance")
plt.plot(dist_trace_sum)
plt.show()

plt.figure()
plt.title("angle")
plt.plot(angle_trace_sum)
plt.show()