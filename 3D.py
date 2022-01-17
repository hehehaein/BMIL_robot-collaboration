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
total_node = 4 #source, n1, n2, destination

class setup :
    def __init__(self, source, destination) :
        self.array = np.empty((total_node,4),int) #source, destination
        self.array[0,:] = source[:]
        self.array[1,:] = destination[:]
        """for i in range (0,total_node-2,1) :
            self.array[i+2,:]= destination[:]"""
        self.array[2,:] = [4,4,4,2]
        self.array[3,:] = [3,3,3,2]

    def do_action(self, i, move):
        next_array = self.array.copy()
        next_array[i][0] = self.array[i][0] + move[0]
        next_array[i][1] = self.array[i][1] + move[1]
        next_array[i][2] = self.array[i][2] + move[2]
        next_array[i][3] = self.array[i][3] + move[3]
        return next_array

    def cal_adjacency(self, next_array):
        adj_array = np.empty((total_node,total_node),bool)
        for i in range (0, total_node, 1):
            for j in range (0,total_node,1):
                distance = math.sqrt(math.pow(next_array[i][0] - next_array[j][0], 2)
                                     + math.pow(next_array[i][1] - next_array[j][1], 2)
                                     + math.pow(next_array[i][2] - next_array[j][2], 2))
                if distance < next_array[i][3]:
                    adj_array[i][j] = True
                else :
                    adj_array[i][j] = False

        return adj_array

    def cal_throughput(self, adj_array):
        graph = nx.Graph()
        for i in range(0,total_node,1):
            graph.add_node(i)
        for i in range(0, total_node, 1) :
            for j in range (0, total_node, 1):
                if adj_array[i][j] :
                    graph.add_edge(i,j)
        #nx.draw(graph)
        #plt.show()

        if nx.has_path(graph,0, 1) :
            path_hop = nx.shortest_path_length(graph,0, 1)
        else :
            path_hop = np.inf

        if path_hop != np.inf:
            throughput = 20/path_hop
        else :
            throughput = 0

        return throughput

    def cal_foot_of_perpendicular(self, next_array, i, move):
        foot_of_perpendicular = math.sqrt(math.pow(((-2*(self.array[i][0] - self.array[0][0])) + (self.array[i][1] - self.array[0][1]) + (self.array[i][2] - self.array[0][2]))/3,2)
                                          +math.pow((((self.array[i][0] - self.array[0][0])) + (-2)*(self.array[i][1] - self.array[0][1]) + (self.array[i][2] - self.array[0][2]))/3,2)
                                          +math.pow((((self.array[i][0] - self.array[0][0])) + (self.array[i][1] - self.array[0][1]) + (-2)*(self.array[i][2] - self.array[0][2]))/3,2))\
                                - math.sqrt(math.pow(((-2*(next_array[i][0] - self.array[0][0])) + (next_array[i][1] - self.array[0][1]) + (next_array[i][2] - self.array[0][2]))/3,2)
                                          +math.pow((((next_array[i][0] - self.array[0][0])) + (-2)*(next_array[i][1] - self.array[0][1]) + (next_array[i][2] - self.array[0][2]))/3,2)
                                          +math.pow((((next_array[i][0] - self.array[0][0])) + (next_array[i][1] - self.array[0][1]) + (-2)*(next_array[i][2] - self.array[0][2]))/3,2))\
                                +move[3]
        return foot_of_perpendicular

    def cal_used_energy_to_move(self,move):
        energy_move = math.sqrt((math.pow(move[0],2) + math.pow(move[1],2) + math.pow(move[2],2)))
        return energy_move

    def cal_used_energy_to_keep_txr(self, i, next_array):
        energy_txr = (next_array[i][3])*(2/5)
        return energy_txr

    def cal_reward(self,throughput, foot_of_perpendicular, energy_move, energy_txr):
        u = 7 #constant that guarantees the reward to be non-negative
        reward = 7 + throughput + foot_of_perpendicular - energy_move - energy_txr
        return reward

    def reset(self,destination):
        for i in range (0,total_node-2,1) :
            self.array[i+2,:]= destination[:]

if __name__ == '__main__':

    source = np.array([0,0,0,2])
    destination = np.array([4,4,4,2])
    i = 2 #action
    move = np.array([-1,-1,0,0]) #action

    env = setup(source, destination)
    next_arr = env.do_action(i, move)
    adj_arr = env.cal_adjacency(next_arr)
    throughput = env.cal_throughput(adj_arr)
    foot = env.cal_foot_of_perpendicular(next_arr, i, move)
    e_move = env.cal_used_energy_to_move(move)
    e_txr = env.cal_used_energy_to_keep_txr(i,next_arr)
    reward = env.cal_reward(throughput,foot,e_move,e_txr)
    #print("throughput : ",throughput)
    #print("foot : " ,foot)
    #print("e_move : " ,e_move)
    #print("e_txr : " ,e_txr)
    print("reward : " ,reward)
    print("next_arr :\n",next_arr)
    #print("array :\n",env.array)

    scatter_array = np.empty((3,total_node),int)
    for i in range(0, 3,1):
        for j in range(0,total_node,1) :
            scatter_array[i][j]=next_arr[j][i]
    print("scatter_array :\n",scatter_array)


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim3d(0, 4)
    ax.set_ylim3d(4, 0)
    ax.set_zlim3d(0, 4)
    # scatter() 함수에 준비된 x, y, z 배열 값을 입력
    # marker = 점의 형태
    # s = 점의 크기
    # c = 점의 색깔
    ax.scatter(scatter_array[0],scatter_array[1],scatter_array[2], marker='o', s=15, c='darkgreen')
    #ax.view_init(40,30) #각도 지정

    plt.show()
    env.reset(destination)





