#%matplotlib notebook #주피터에서 마우스로 그래프움직일때 사용
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt, animation
from mpl_toolkits.mplot3d import Axes3D
import time
import math


"""
basic setup
4*4*4 3D space
source, destination's transmission radius = 2

observation : 
action : move the node, change the transmission radius
reward : throughput + 수선의 발 길이의 차 - node 이동거리 - transmission radius 유지를 위한 에너지량
"""

#source, n1, n2, destination
total_node = 4
MAX_LOCATION = 4  #max location
N=2

class setup :
    def __init__(self, source, destination) :
        self.state_array = np.empty((total_node, 4), int) #source, destination
        self.state_array[0, :] = [1, 4, 1, 3]
        self.state_array[1, :] = [2, 3, 3, 3]
        """for i in range (0,total_node-2,1) :
            self.state_array[i+2,:]= destination[:]"""
        self.state_array[2, :] = source[:]
        self.state_array[3, :] = destination[:]

    #action 수행하기
    def do_action(self, i, move):
        next_state_array = self.state_array.copy()
        next_state_array[i][0] = self.state_array[i][0] + move[0] #x축
        next_state_array[i][1] = self.state_array[i][1] + move[1] #y축
        next_state_array[i][2] = self.state_array[i][2] + move[2] #z축
        next_state_array[i][3] = self.state_array[i][3] + move[3] #txr

        if ((next_state_array[i][2]>height_max) or (next_state_array[i][2]<height_min)) :
            next_state_array[i][2] = self.state_array[i][2]

        return next_state_array

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
            # print('tmp_array\n', tmp2_array)
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
            h = math.sqrt(
                math.pow(constant * kx - x, 2) + math.pow(constant * ky - y, 2) + math.pow(constant * kz - z, 2))
            return h

        # (시간t일때의 수선의 발 - 시간t+1일때 수선의 발)길이 구하기
        def cal_foot_of_perpendicular(self, state_array, next_state_array, source, destination, i):
            foot_of_perpendicular = self.cal_h(state_array[i][0], state_array[i][1], state_array[i][2], source,
                                               destination) \
                                    - self.cal_h(next_state_array[i][0], next_state_array[i][1], next_state_array[i][2],
                                                 source, destination) \
                                    - state_array[i][3] + next_state_array[i][3]
            return foot_of_perpendicular

        def cal_foot(self, next_state_array, source, destination, i):
            foot = next_state_array[i][3] - self.cal_h(next_state_array[i][0], next_state_array[i][1], next_state_array[i][2], source, destination)
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
        energy_move = 11.2 * (0.5 * (x + y) + z)
        return energy_move

    def cal_used_energy_to_keep_txr(self, my_txr):
        energy_txr = math.pow(my_txr, 2)
        return energy_txr

    def cal_reward(self, throughput, foot_of_perpendicular, dispersed, energy_move, energy_txr):
        u = 5  # constant that guarantees the reward to be non-negative
        reward = u + (throughput) + (foot_of_perpendicular) + (dispersed) - (energy_move / 22) - (energy_txr / 2)
        return reward

    #에피소드마다 node들의 위치 리셋시키기
    def reset(self,destination):
        for i in range (0,total_node-2,1) :
            self.state_array[i + 2, :]= destination[:]



#메인함수
if __name__ == '__main__':

    source = np.array([0,0,0,2])
    destination = np.array([4,4,4,0])
    height_max = 4
    height_min = 1

    i = 0 #action
    move = np.array([0,0,0,0]) #action

    env = setup(source, destination)
    next_arr = env.do_action(i, move)
    adj_arr = env.cal_adjacency(next_arr)
    throughput = env.cal_throughput(adj_arr)
    dispersed = env.cal_dispersed(i, next_arr[i][3], adj_arr)
    foot = env.cal_foot(next_arr, source, destination, i)
    e_move = env.cal_used_energy_to_move(move)
    e_txr = env.cal_used_energy_to_keep_txr(i,next_arr)
    reward = env.cal_reward(throughput, dispersed, foot, e_move, e_txr)

    print("move : ",move)
    print("throughput : ",throughput)

    print("dispersed : ",dispersed)
    print("foot : " ,foot)
    #print("e_move : " ,e_move)
    #print("e_txr : " ,e_txr)
    print("reward : " ,reward)
    print("array :\n", env.state_array)
    print("next_arr :\n",next_arr)
    print("adj_arr : \n",adj_arr)


    #node들의 좌표 배열 구하기
    scatter_array = np.empty((4,total_node),int)
    for i in range(0, 4,1):
        for j in range(0,total_node,1) :
            scatter_array[i][j]=next_arr[j][i]
    #print("scatter_array :\n",scatter_array)

    '''def create_sphere(cx, cy, cz, r):

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
    ax.set_xlim3d(0 - 2, MAX_LOCATION + 2)
    ax.set_ylim3d(MAX_LOCATION + 2, 0 - 2)
    ax.set_zlim3d(0 - 2, MAX_LOCATION + 2)

    color_list = ("red","orange","green","blue","purple","black")
    scatter_array=np.transpose(next_arr)  #x, y, z값들을 행 별로 묶기

    for i in range(0,total_node,1):
        (x,y,z) = create_sphere(next_arr[i][0], next_arr[i][1],next_arr[i][2],next_arr[i][3])
        ax.auto_scale_xyz([0, 500], [0, 500], [0, 0.15])
        ax.plot_surface(x,y,z,color=color_list[i],linewidth=0,alpha=0.3)
        ax.scatter(scatter_array[1],scatter_array[1],scatter_array[2], marker='o', s=80, c='darkgreen')'''

    plt.show()

    env.reset(destination)