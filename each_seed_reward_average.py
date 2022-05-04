import gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import seaborn as sns;
import pickle
import matplotlib.lines as mlines
from statistics import NormalDist

NUM_EPISODES = 200000
STEPS = 20
iter = 10

rewards = []
with open("rewards_0.pickle","rb") as f:
    rewards.append(pickle.load(f))
with open("rewards_1.pickle","rb") as f:
    rewards.append(pickle.load(f))
with open("rewards_2.pickle","rb") as f:
    rewards.append(pickle.load(f))
with open("rewards_3.pickle","rb") as f:
    rewards.append(pickle.load(f))
with open("rewards_4.pickle","rb") as f:
    rewards.append(pickle.load(f))
with open("rewards_5.pickle","rb") as f:
    rewards.append(pickle.load(f))
with open("rewards_6.pickle","rb") as f:
    rewards.append(pickle.load(f))
with open("rewards_7.pickle","rb") as f:
    rewards.append(pickle.load(f))
with open("rewards_8.pickle","rb") as f:
    rewards.append(pickle.load(f))
with open("rewards_9.pickle","rb") as f:
    rewards.append(pickle.load(f))

#with open("rewards_40_0.pickle","rb") as f:
#     rewards.append(pickle.load(f))
# with open("rewards_40_1.pickle","rb") as f:
#     rewards.append(pickle.load(f))
# with open("rewards_40_2.pickle","rb") as f:
#     rewards.append(pickle.load(f))
# with open("rewards_40_3.pickle","rb") as f:
#     rewards.append(pickle.load(f))
# with open("rewards_40_4.pickle","rb") as f:
#     rewards.append(pickle.load(f))
# with open("rewards_40_5.pickle","rb") as f:
#     rewards.append(pickle.load(f))
# with open("rewards_40_6.pickle","rb") as f:
#     rewards.append(pickle.load(f))
# with open("rewards_40_7.pickle","rb") as f:
#     rewards.append(pickle.load(f))
# with open("rewards_40_8.pickle","rb") as f:
#     rewards.append(pickle.load(f))
# with open("rewards_40_9.pickle","rb") as f:
#     rewards.append(pickle.load(f))

throughputs = []
with open("throughputs_0.pickle","rb") as f:
    throughputs.append(pickle.load(f))
with open("throughputs_1.pickle","rb") as f:
    throughputs.append(pickle.load(f))
with open("throughputs_2.pickle","rb") as f:
    throughputs.append(pickle.load(f))
with open("throughputs_3.pickle","rb") as f:
    throughputs.append(pickle.load(f))
with open("throughputs_4.pickle","rb") as f:
    throughputs.append(pickle.load(f))
with open("throughputs_5.pickle","rb") as f:
    throughputs.append(pickle.load(f))
with open("throughputs_6.pickle","rb") as f:
    throughputs.append(pickle.load(f))
with open("throughputs_7.pickle","rb") as f:
    throughputs.append(pickle.load(f))
with open("throughputs_8.pickle","rb") as f:
    throughputs.append(pickle.load(f))
with open("throughputs_9.pickle","rb") as f:
    throughputs.append(pickle.load(f))

# with open("throughputs_40_0.pickle","rb") as f:
#     throughputs.append(pickle.load(f))
# with open("throughputs_40_1.pickle","rb") as f:
#     throughputs.append(pickle.load(f))
# with open("throughputs_40_2.pickle","rb") as f:
#     throughputs.append(pickle.load(f))
# with open("throughputs_40_3.pickle","rb") as f:
#     throughputs.append(pickle.load(f))
# with open("throughputs_40_4.pickle","rb") as f:
#     throughputs.append(pickle.load(f))
# with open("throughputs_40_5.pickle","rb") as f:
#     throughputs.append(pickle.load(f))
# with open("throughputs_40_6.pickle","rb") as f:
#     throughputs.append(pickle.load(f))
# with open("throughputs_40_7.pickle","rb") as f:
#     throughputs.append(pickle.load(f))
# with open("throughputs_40_8.pickle","rb") as f:
#     throughputs.append(pickle.load(f))
# with open("throughputs_40_9.pickle","rb") as f:
#     throughputs.append(pickle.load(f))

def get_mean(array, k):
    means = []
    m = k   # episode를 몇개씩 묶을건지
    if k == STEPS:
        m = 1   # 1개의 episode의 평균 구하기
    for n in range(0, NUM_EPISODES // m, 1):
        sum = 0
        for i in range(0, k, 1):
            sum += array[(n * k) + i]
        means.append(sum / k)
    return means


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


reward_means = np.reshape(rewards, (iter,NUM_EPISODES*STEPS))  #각 iter마다의 reward로 2차원 배열을 만들기
#한 iter에서 하나의 episode의 평균으로 묶음
reward_means2 = []
for i in range(iter):
    print('1:', max(reward_means[i]))
    tmp = get_mean(reward_means[i], STEPS)
    reward_means2.append(tmp)

#한 iter에서 100개의 episode의 평균으로 묶음
reward_means2_100 = []
for i in range(iter):
    print('100:',max(reward_means2[i]))
    tmp = get_mean(reward_means2[i], 100)
    reward_means2_100.append(tmp)
#dataframe별로 표준편차 나타낼려고 배열 모양 변형
reward_means2 = np.transpose(reward_means2)
reward_means2 = reward_means2.flatten()
reward_means2_100 = np.transpose(reward_means2_100)
reward_means2_100 = reward_means2_100.flatten()

throughput_means = np.reshape(throughputs, (iter,NUM_EPISODES*STEPS))  #각 iter마다의 reward로 2차원 배열을 만들기
#한 iter에서 하나의 episode의 평균으로 묶음
throughput_means2 = []
for i in range(iter):
    tmp = get_mean(throughput_means[i], STEPS)
    throughput_means2.append(tmp)

#1개의 에피소드 평균 throughput을 hit ratio로 변환
for i in range(iter):
    for j in range(NUM_EPISODES):
        throughput_means2[i][j] /= STEPS

#한 iter에서 100개의 episode의 평균으로 묶음
throughput_means2_100 = []
for i in range(iter):
    tmp = get_mean(throughput_means2[i], 100)
    throughput_means2_100.append(tmp)
#dataframe별로 표준편차 나타낼려고 배열 모양 변형
throughput_means2 = np.transpose(throughput_means2)
throughput_means2 = throughput_means2.flatten()
throughput_means2_100 = np.transpose(throughput_means2_100)
throughput_means2_100 = throughput_means2_100.flatten()

print('max:',max(reward_means2))
print('max_100:',max(reward_means2_100))
#츌력
plt.figure()
sns.set_style('darkgrid')
d = {'100 episode': make_list(NUM_EPISODES/100*iter, iter),
     'reward': reward_means2_100,
     'hit ratio': throughput_means2_100}
df = pd.DataFrame(data=d)
reward = sns.lineplot(data=df, x='100 episode', y='reward', color='blue', ci=68)
reward.set(title='reward')
plt.axhline(11.666666666666668, c='black', ls ='--')
mark_reward = mlines.Line2D([], [], color='blue', linestyle='-', label='reward')
mark_upper = mlines.Line2D([], [], color='black', linestyle='--', label='upper bound')

plt.legend(handles=[mark_reward, mark_upper], loc='lower right')

last = []
for i in range(10):
    last.append((reward_means[i][120000*STEPS-1]))
print(last)
print(np.std(last))
print(np.mean(last))
# plt.figure()
# reward = sns.lineplot(data=df, x='1 episode', y='hit ratio',ci='sd')
# reward.set(title='hit ratio')
# plt.figure()
# d = {'1 episode': make_list(NUM_EPISODES*iter, iter),
#      'reward': reward_means2,
#      'hit ratio': throughput_means2}
# df = pd.DataFrame(data=d)
# fig, axe1 = plt.subplots()
# axe2 = axe1.twinx()
# hit_ratio = sns.lineplot(ax=axe1, data=df, x='1 episode', y='hit ratio', color='red')
# reward = sns.lineplot(ax=axe2, data=df, x='1 episode', y='reward', color='blue')
# axe1.legend(['hit ratio','reward'])
# axe1.set_ylabel('hit ratio',fontsize=14)
# axe2.set_ylabel('reward',fontsize=14)

plt.show()