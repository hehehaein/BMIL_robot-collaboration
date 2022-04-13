# -*- coding: utf-8 -*-
import gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import seaborn as sns;

sns.set()
import json
import pickle
from collections import namedtuple, deque, OrderedDict
from ray.tune.registry import register_env

from gymExample.gym_example.envs.my_dqn_env import My_DQN
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

select_env = "dqn-v0"
register_env(select_env, lambda config: My_DQN())
env = gym.make(select_env).unwrapped

# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

my_seed = 1
random.seed(my_seed)
np.random.seed(my_seed)
torch.manual_seed(my_seed)
torch.backends.cudnn.deterministic = True
os.environ["PYTHONHASHSEED"] = str(my_seed)

# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


HE = 'kaiming_uniform'
ACTIV = 'LeakyRelu0.1'


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(16, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 81)
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear3.weight, mode='fan_in', nonlinearity='leaky_relu')

    # 최적화 중에 다음 행동을 결정하기 위해서 하나의 요소 또는 배치를 이용해 호촐됩니다.
    # ([[left0exp,right0exp]...]) 를 반환합니다.
    def forward(self, x):
        # x = x.to(device)
        # print('forward\n',x)
        m = nn.LeakyReLU(0.1)
        x = m(self.linear1(x))
        x = m(self.linear2(x))
        '''x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))'''
        return self.linear3(x)


BATCH_SIZE = 64
NUM_EPISODES = 400000
STEPS = 20
DISCOUNT_FACTOR = 0.9
EPS_START = 0.99
EPS_END = 0.01
EPS_DECAY = (EPS_START - EPS_END) / (NUM_EPISODES * STEPS * 0.7)
TARGET_UPDATE = 10
UPDATE_FREQ = 1
BUFFER = 100000
LEARNING_RATE = 1e-4
IS_DOUBLE_Q = True
ZERO = False
SCHEDULER = False
SCHEDULER_GAMMA = 0.95

now = time.localtime()
file_name = '{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_1th_{12}-{13}-{14}-{15}'.format(
    NUM_EPISODES, STEPS, DISCOUNT_FACTOR, EPS_DECAY, TARGET_UPDATE, UPDATE_FREQ, '1th',
    BUFFER, LEARNING_RATE, my_seed, IS_DOUBLE_Q, ZERO, SCHEDULER, now.tm_hour, now.tm_min, now.tm_sec)
path = os.path.join(os.getcwd(), 'results')

n_actions = env.action_space.n

policy_net = DQN().to(device)
target_net = DQN().to(device)
tmp_net = DQN().to(device)

target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(BUFFER)
'''if SCHEDULER:
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=SCHEDULER_GAMMA)'''
'''earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=True, mode='min')
trainer = Trainer(callbacks=[earlystop])'''

epslions = []
steps_done = 0


def select_action(state):
    global steps_done
    # 멈춰있는 action 확률 키우기
    a1 = []  # 빈 리스트 생성
    a2 = []
    a3 = []
    for i in range(n_actions):
        a1.append(1)
        a2.append(3 / 356)
        a3.append(0.0044)
    a2[40] = 1 / 16
    a3[40] = 81 / 125

    sample = random.random()
    eps_threshold = max(EPS_END, EPS_START - (EPS_DECAY * steps_done))
    epslions.append(eps_threshold)
    steps_done += 1

    # eps greedy method
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max (1)은 각 행의 가장 큰 열 값을 반환합니다.
            # 최대 결과의 두번째 열은 최대 요소의 주소값이므로,
            # 기대 보상이 더 큰 행동을 선택할 수 있습니다.
            return policy_net(state).max(-1)[1].view(1, 1)
    else:
        if ZERO:  # action이 0일 확률을 키우는 경우
            if (20 * STEPS * NUM_EPISODES) * 0.5 * 0.66 < steps_done:
                return torch.tensor([random.choices(range(0, n_actions), weights=a3)], device=device, dtype=torch.long)
            elif (20 * STEPS * NUM_EPISODES) * 0.5 * 0.33 < steps_done:
                return torch.tensor([random.choices(range(0, n_actions), weights=a2)], device=device, dtype=torch.long)
            else:
                return torch.tensor([random.choices(range(0, n_actions), weights=a1)], device=device, dtype=torch.long)
        else:  # 모두 같은 확률일 경우
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


losses = []
'''scheduler_lrs = []
scheduler_check = False'''


# 최적화의 한 단계를 수행함
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    # initialize
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    index = torch.zeros(BATCH_SIZE, device=device
                        )

    # batch-array의 Transitions을 Transition의 batch-arrays로 전환
    # Changing Format
    # From [[S1, A1, R1, S'1], [S2, A2, R2, S'2], ..., [Sn, An, Rn, S'n]]
    # To [S1, S2, ..., Sn], [A1, A2, ..., An], ..., [S'1, S'2, ...,S'n]
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device,
                                  dtype=torch.bool)

    # next state
    for s in batch.next_state:
        if s is not None:
            non_final_next_states = torch.stack(tuple(torch.Tensor(s)))
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).reshape(BATCH_SIZE, 1, 16)
    # state, action, reward
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Q(s_t, a) 계산
    # state_action_values = policy_net(state_batch).gather(-1, action_batch.squeeze(1))
    state_action_values = policy_net(state_batch).reshape(BATCH_SIZE, 81, 1).gather(1, action_batch)

    # print(state_action_values.size())
    if IS_DOUBLE_Q:
        index = policy_net(non_final_next_states).max(-1)[1].detach()
        next_state_values = target_net(non_final_next_states)[-1][0][index].detach()
    else:
        next_state_values = target_net(non_final_next_states).max(-1)[0].detach()
    # 기대 Q 값 계산
    expected_state_action_values = (next_state_values * DISCOUNT_FACTOR) + reward_batch
    # print('e',expected_state_action_values.unsqueeze(1).size())
    # Huber 손실 계산
    '''loss 계산'''
    criterion = nn.SmoothL1Loss()
    #    loss = criterion(state_action_values, expected_state_action_values)
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 모델 최적화
    optimizer.zero_grad()
    '''backward propagation'''
    loss.backward()
    optimizer.step()  # 경사하강법(gradient descent). 옵티마이저는 .grad 에 저장된 변화도(gradients)에 따라 각 매개변수를 조정(adjust)합니다.
    losses.append(loss.item())


'''if scheduler_check: #scheduler 실행, 값 그래프 뽑기
        scheduler.step()
        scheduler_lrs.append(scheduler.get_lr())
    else:
        scheduler_lrs.append(LEARNING_RATE)'''

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

actions = []
throughputs = []
foots = []
disperses = []
moves = []
txrs = []
rewards = []
throughput_counts = []  # 한 에피소드당 throughput 연결횟수
stay_count = []  # 한 에피소드당 optimal 위치에 머무는 횟수
opti_count = []  # 한 에피소드당 optimal한 위치에 가는 횟수
'''scatters_front = []
scatters_middle = []
scatters_tail = []'''
z_throughput = np.zeros((4, 5, 5))  # 한 에피소드당 throughput값를 z에 대한 시작위치에다가 1 저장
z_throughput_count = np.zeros((4, 5, 5))  # throughput이 연결되는 z에 대한 해당위치에 몇변 가는지
z_txr_optimal = np.zeros((16, 5, 5))  # s_(t+1)이 optimal한 위치이면 count
z_txr_visit = np.zeros((16, 5, 5))
z_txr_reward = np.zeros((16, 5, 5))  # reward plot
distribution = np.zeros((4, 5, 5))  # 시작위치가 골고루 분포해서 생성되는지 확인
reward_count = 0
optimal = 0
visit_reward_count = 0
iter = 10
for i in range(iter):
    my_seed = i
    random.seed(my_seed)
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(my_seed)
    each_rewards = []

    for i_episode in tqdm(range(NUM_EPISODES)):
        throughput_count = 0  # 한 에피소드당 throghput이 연결되는 횟수
        # optimal = 0
        stay = 0
        move_count = 0  # 몇 스텝 갔는지 확인용
        # 환경과 상태 초기화
        state = env.reset()
        state = torch.Tensor(state)

        # distribution print
        distribution[state[2].int().item() - 1][state[0].int().item()][state[1].int().item()] += 1
        state_for_save = state

        if SCHEDULER and i_episode == NUM_EPISODES // 2:
            scheduler_check = True
        for t in range(0, STEPS, 1):
            move_count += 1

            # 행동 선택과 수행
            action = select_action(state)
            actions.append(action.item())
            next_state, reward, done, last_set = env.step(action.item())

            state_reshape = np.reshape(state, (env.N + 2, 4))
            next_state_reshape = np.reshape(next_state, (env.N + 2, 4))

            if last_set[0] != 0:
                print(i, state_reshape[0], next_state_reshape[0],
                      "action:%2d%2d%2d%2d reward:%.6f count:%2d"
                      % (last_set[5], last_set[6], last_set[7], last_set[8], reward, move_count))

            if i_episode >= (NUM_EPISODES - 2):
                print(state_reshape[0], next_state_reshape[0],
                      "action:%2d%2d%2d%2d throughput:%6.3f foot:%6.3f txr:%3d"  # dispersed:%6.3f move:%6.3f
                      % (last_set[5], last_set[6], last_set[7], last_set[8],
                         last_set[0], last_set[1], last_set[4]))  # last_set[2], last_set[3],

            if last_set[0] != 0:
                throughput_count += 1
            throughputs.append(last_set[0])

            rewards.append(reward)
            each_rewards.append(reward)
            reward = torch.tensor([reward], device=device)

            next_state = torch.Tensor(next_state)
            if done:
                next_state = None
            # 메모리에 변이 저장
            memory.push(state, action, next_state, reward)
            # 다음 상태로 이동
            state = next_state

        # (정책 네트워크에서) 최적화 한단계 수행
        if i_episode % UPDATE_FREQ == 0 and i_episode != NUM_EPISODES - 1:
            optimize_model()
            if done:
                # episode_durations.append(t + 1)
                # plot_durations()
                break

        # gradient 출력
        if i_episode == NUM_EPISODES - 1:
            for p in policy_net.parameters():
                with torch.no_grad():
                    print(p.grad, len(p.grad), p.grad.shape)

        # 목표 네트워크 업데이트, 모든 웨이트와 바이어스 복사
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # stay_count.append(stay)
        throughput_counts.append(throughput_count)

    plt.figure()
    reward_means2 = get_mean(each_rewards, STEPS)
    reward_means2_100 = []
    reward_means2_100 = get_mean(reward_means2, 100)
    d = {'1 episode': make_list(NUM_EPISODES, 1),
         'reward': reward_means2}
    df = pd.DataFrame(data=d)
    reward = sns.lineplot(data=df, x='1 episode', y='reward', ci='sd')
    reward.set(title='reward in {}'.format(i))

    plt.figure()
    d = {'100 episode': make_list(NUM_EPISODES // 100, 1),
         'reward': reward_means2_100}
    df = pd.DataFrame(data=d)
    reward = sns.lineplot(data=df, x='100 episode', y='reward', ci='sd')
    reward.set(title='reward in {}'.format(i))

print('Complete')

plt.figure()
reward_means = np.reshape(rewards, (iter,NUM_EPISODES*STEPS))  #각 iter마다의 reward로 2차원 배열을 만들기

#한 iter에서 하나의 episode의 평균으로 묶음
reward_means2 = []
for i in range(iter):
    tmp = get_mean(reward_means[i], STEPS)
    reward_means2.append(tmp)

#한 iter에서 100개의 episode의 평균으로 묶음
reward_means2_100 = []
for i in range(iter):
    tmp = get_mean(reward_means2[i], 100)
    reward_means2_100.append(tmp)

#dataframe별로 표준편차 나타낼려고 배열 모양 변형
reward_means2 = np.transpose(reward_means2)
reward_means2 = reward_means2.flatten()
reward_means2_100 = np.transpose(reward_means2_100)
reward_means2_100 = reward_means2_100.flatten()

#츌력
d = {'1 episode': make_list(NUM_EPISODES*iter, iter),
     'reward': reward_means2}
df = pd.DataFrame(data=d)
reward = sns.lineplot(data=df, x='1 episode', y='reward', ci='sd')
reward.set(title='reward')

plt.figure()
d = {'100 episode': make_list(NUM_EPISODES//100*iter, iter),
     'reward': reward_means2_100}
df = pd.DataFrame(data=d)
reward = sns.lineplot(data=df, x='100 episode', y='reward', ci='sd')
reward.set(title='reward')

print('seed  ', my_seed)
print('BUFFER  ', BUFFER)
print('ZERO  ', ZERO)
print('IS_DOUBLE  ', IS_DOUBLE_Q)
print('SCHEDULER  ', SCHEDULER)
print('STEPS  ', STEPS)
print('EPISODES  ', NUM_EPISODES)
print('EPS_DECAY  ', EPS_DECAY)
print('UPDATE_FREQ  ', UPDATE_FREQ)
print('TARGET_UPDATE  ', TARGET_UPDATE)
print('DISCOUNT_FACTOR  ', DISCOUNT_FACTOR)
print('LEARNING_RATE  ', LEARNING_RATE)
print('source txr = 1, MIN_HEIGHT = 0')

env.close()
plt.ioff()
plt.show()