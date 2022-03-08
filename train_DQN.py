# -*- coding: utf-8 -*-
"""
강화 학습 (DQN) 튜토리얼
=====================================
**Author**: `Adam Paszke <https://github.com/apaszke>`_
  **번역**: `황성수 <https://github.com/adonisues>`_

이 튜토리얼에서는 `OpenAI Gym <https://gym.openai.com/>`__ 의
CartPole-v0 태스크에서 DQN (Deep Q Learning) 에이전트를 학습하는데
PyTorch를 사용하는 방법을 보여드립니다.

**태스크**

에이전트는 연결된 막대가 똑바로 서 있도록 카트를 왼쪽이나 오른쪽으로
움직이는 두 가지 동작 중 하나를 선택해야 합니다.
다양한 알고리즘과 시각화 기능을 갖춘 공식 순위표를
`Gym 웹사이트 <https://gym.openai.com/envs/CartPole-v0>`__ 에서 찾을 수 있습니다.

.. figure:: /_static/img/cartpole.gif
   :alt: cartpole

   cartpole

에이전트가 현재 환경 상태를 관찰하고 행동을 선택하면,
환경이 새로운 상태로 *전환* 되고 작업의 결과를 나타내는 보상도 반환됩니다.
이 태스크에서 매 타임스텝 증가마다 보상이 +1이 되고, 만약 막대가 너무 멀리
떨어지거나 카트가 중심에서 2.4 유닛 이상 멀어지면 환경이 중단됩니다.
이것은 더 좋은 시나리오가 더 오랫동안 더 많은 보상을 축적하는 것을 의미합니다.

카트폴 태스크는 에이전트에 대한 입력이 환경 상태(위치, 속도 등)를 나타내는
4개의 실제 값이 되도록 설계되었습니다. 그러나 신경망은 순수하게 그 장면을 보고
태스크를 해결할 수 있습니다 따라서 카트 중심의 화면 패치를 입력으로 사용합니다.
이 때문에 우리의 결과는 공식 순위표의 결과와 직접적으로 비교할 수 없습니다.
우리의 태스크는 훨씬 더 어렵습니다.
불행히도 모든 프레임을 렌더링해야되므로 이것은 학습 속도를 늦추게됩니다.

엄밀히 말하면, 현재 스크린 패치와 이전 스크린 패치 사이의 차이로 상태를 표시할 것입니다.
이렇게하면 에이전트가 막대의 속도를 한 이미지에서 고려할 수 있습니다.

**패키지**

먼저 필요한 패키지를 가져옵니다. 첫째, 환경을 위해
`gym <https://gym.openai.com/docs>`__ 이 필요합니다.
(`pip install gym` 을 사용하여 설치하십시오).
또한 PyTorch에서 다음을 사용합니다:

-  신경망 (``torch.nn``)
-  최적화 (``torch.optim``)
-  자동 미분 (``torch.autograd``)
-  시각 태스크를 위한 유틸리티들 (``torchvision`` - `a separate
   package <https://github.com/pytorch/vision>`__).

"""

import gym
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import time
import os

from collections import namedtuple, deque
from itertools import count
from ray.tune.registry import register_env

from gymExample.gym_example.envs.my_dqn_env import My_DQN

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

seed = 3
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
os.environ["PYTHONHASHSEED"]=str(seed)



# GPU를 사용할 경우
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
######################################################################
# 재현 메모리(Replay Memory)
# -------------------------------
#
# 우리는 DQN 학습을 위해 경험 재현 메모리를 사용할 것입니다.
# 에이전트가 관찰한 전환(transition)을 저장하고 나중에 이 데이터를
# 재사용할 수 있습니다. 무작위로 샘플링하면 배치를 구성하는 전환들이
# 비상관(decorrelated)하게 됩니다. 이것이 DQN 학습 절차를 크게 안정시키고
# 향상시키는 것으로 나타났습니다.
#
# 이를 위해서 두개의 클래스가 필요합니다:
#
# -  ``Transition`` - 우리 환경에서 단일 전환을 나타내도록 명명된 튜플.
#    그것은 화면의 차이인 state로 (state, action) 쌍을 (next_state, reward) 결과로 매핑합니다.
# -  ``ReplayMemory`` - 최근 관찰된 전이를 보관 유지하는 제한된 크기의 순환 버퍼.
#    또한 학습을 위한 전환의 무작위 배치를 선택하기위한
#    ``.sample ()`` 메소드를 구현합니다.

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


######################################################################
# 이제 모델을 정의합시다. 그러나 먼저 DQN이 무엇인지 간단히 요약해 보겠습니다.
#
# DQN 알고리즘
# -------------
#
# 우리의 환경은 결정론적이므로 여기에 제시된 모든 방정식은 단순화를 위해
# 결정론적으로 공식화됩니다. 강화 학습 자료은 환경에서 확률론적 전환에
# 대한 기대값(expectation)도 포함할 것입니다.
#
# 우리의 목표는 할인된 누적 보상 (discounted cumulative reward)을
# 극대화하려는 정책(policy)을 학습하는 것입니다.
# :math:`R_{t_0} = \sum_{t=t_0}^{\infty} \gamma^{t - t_0} r_t`, 여기서
# :math:`R_{t_0}` 는 *반환(return)* 입니다. 할인 상수,
# :math:`\gamma`, 는 :math:`0` 과 :math:`1` 의 상수이고 합계가
# 수렴되도록 보장합니다. 에이전트에게 불확실한 먼 미래의 보상이
# 가까운 미래의 것에 비해 덜 중요하게 만들고, 이것은 상당히 합리적입니다.
#
# Q-learning의 주요 아이디어는 만일 함수 :math:`Q^*: State \times Action \rightarrow \mathbb{R}` 를
# 가지고 있다면 반환이 어떻게 될지 알려줄 수 있고,
# 만약 주어진 상태(state)에서 행동(action)을 한다면, 보상을 최대화하는
# 정책을 쉽게 구축할 수 있습니다:
#
# .. math:: \pi^*(s) = \arg\!\max_a \ Q^*(s, a)
#
# 그러나 세계(world)에 관한 모든 것을 알지 못하기 때문에,
# :math:`Q^*` 에 도달할 수 없습니다. 그러나 신경망은
# 범용 함수 근사자(universal function approximator)이기 때문에
# 간단하게 생성하고 :math:`Q^*` 를 닮도록 학습할 수 있습니다.
#
# 학습 업데이트 규칙으로, 일부 정책을 위한 모든 :math:`Q` 함수가
# Bellman 방정식을 준수한다는 사실을 사용할 것입니다:
#
# .. math:: Q^{\pi}(s, a) = r + \gamma Q^{\pi}(s', \pi(s'))
#
# 평등(equality)의 두 측면 사이의 차이는
# 시간차 오류(temporal difference error), :math:`\delta` 입니다.:
#
# .. math:: \delta = Q(s, a) - (r + \gamma \max_a Q(s', a))
#
# 오류를 최소화하기 위해서 `Huber
# loss <https://en.wikipedia.org/wiki/Huber_loss>`__ 를 사용합니다.
# Huber loss 는 오류가 작으면 평균 제곱 오차( mean squared error)와 같이
# 동작하고 오류가 클 때는 평균 절대 오류와 유사합니다.
# - 이것은 :math:`Q` 의 추정이 매우 혼란스러울 때 이상 값에 더 강건하게 합니다.
# 재현 메모리에서 샘플링한 전환 배치 :math:`B` 에서 이것을 계산합니다:
#
# .. math::
#
#    \mathcal{L} = \frac{1}{|B|}\sum_{(s, a, s', r) \ \in \ B} \mathcal{L}(\delta)
#
# .. math::
#
#    \text{where} \quad \mathcal{L}(\delta) = \begin{cases}
#      \frac{1}{2}{\delta^2}  & \text{for } |\delta| \le 1, \\
#      |\delta| - \frac{1}{2} & \text{otherwise.}
#    \end{cases}
#
# Q-네트워크
# ^^^^^^^^^^^
#
# 우리 모델은 현재와 이전 스크린 패치의 차이를 취하는
# CNN(convolutional neural network) 입니다. 두가지 출력 :math:`Q(s, \mathrm{left})` 와
# :math:`Q(s, \mathrm{right})` 가 있습니다. (여기서 :math:`s` 는 네트워크의 입력입니다)
# 결과적으로 네트워크는 주어진 현재 입력에서 각 행동의 *기대값* 을 예측하려고 합니다.
#

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


######################################################################
# 학습
# --------
#
# 하이퍼 파라미터와 유틸리티
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 이 셀은 모델과 최적화기를 인스턴스화하고 일부 유틸리티를 정의합니다:
#
# -  ``select_action`` - Epsilon Greedy 정책에 따라 행동을 선택합니다.
#    간단히 말해서, 가끔 모델을 사용하여 행동을 선택하고 때로는 단지 하나를
#    균일하게 샘플링할 것입니다. 임의의 액션을 선택할 확률은
#    ``EPS_START`` 에서 시작해서 ``EPS_END`` 를 향해 지수적으로 감소할 것입니다.
#    ``EPS_DECAY`` 는 감쇠 속도를 제어합니다.
# -  ``plot_durations`` - 지난 100개 에피소드의 평균(공식 평가에서 사용 된 수치)에 따른
#    에피소드의 지속을 도표로 그리기 위한 헬퍼. 도표는 기본 훈련 루프가
#    포함 된 셀 밑에 있으며, 매 에피소드마다 업데이트됩니다.
#
BATCH_SIZE = 32
num_episodes = 8000
DISCOUNT_FACTOR = 0.9
EPS_START = 0.99
EPS_END = 0.1
EPS_DECAY = 0.00001255
TARGET_UPDATE = 1

now = time.localtime()
#str = 'file{0}_{1}_{2}_{3}_{4}_{5}'.format(
#    now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
str = 'test{0}-{1}-{2}_{3}_{4}_{5}_{6}_{7}_{8}'.format(
    num_episodes, DISCOUNT_FACTOR, '1e-6','1th','txr2', seed, now.tm_hour, now.tm_min, now.tm_sec)
path = os.path.join(os.getcwd(), 'results')

# AI gym에서 반환된 형태를 기반으로 계층을 초기화 하도록 화면의 크기를
# 가져옵니다. 이 시점에 일반적으로 3x40x90 에 가깝습니다.
# 이 크기는 get_screen()에서 고정, 축소된 렌더 버퍼의 결과입니다.
# init_screen = get_screen()
# _, _, screen_height, screen_width = init_screen.shape

# gym 행동 공간에서 행동의 숫자를 얻습니다.
n_actions = env.action_space.n

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=1e-6)
memory = ReplayMemory(100000)

steps_done = 0
epslions = []


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = max(EPS_END, EPS_START - (EPS_DECAY * steps_done))
    epslions.append(eps_threshold)

    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max (1)은 각 행의 가장 큰 열 값을 반환합니다.
            # 최대 결과의 두번째 열은 최대 요소의 주소값이므로,
            # 기대 보상이 더 큰 행동을 선택할 수 있습니다.
            return policy_net(state).max(-1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


######################################################################
# 학습 루프
# ^^^^^^^^^^^^^
#
# 최종적으로 모델 학습을 위한 코드.
#
# 여기서, 최적화의 한 단계를 수행하는 ``optimize_model`` 함수를 찾을 수 있습니다.
# 먼저 배치 하나를 샘플링하고 모든 Tensor를 하나로 연결하고
# :math:`Q(s_t, a_t)` 와  :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)` 를 계산하고
# 그것들을 손실로 합칩니다. 우리가 설정한 정의에 따르면 만약 :math:`s` 가
# 마지막 상태라면 :math:`V(s) = 0` 입니다.
# 또한 안정성 추가 위한 :math:`V(s_{t+1})` 계산을 위해 목표 네트워크를 사용합니다.
# 목표 네트워크는 대부분의 시간 동결 상태로 유지되지만, 가끔 정책
# 네트워크의 가중치로 업데이트됩니다.
# 이것은 대개 설정한 스텝 숫자이지만 단순화를 위해 에피소드를 사용합니다.
#

losses = []
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). 이것은 batch-array의 Transitions을 Transition의 batch-arrays로
    # 전환합니다.
    batch = Transition(*zip(*transitions))
    # 최종이 아닌 상태의 마스크를 계산하고 배치 요소를 연결합니다
    # (최종 상태는 시뮬레이션이 종료 된 이후의 상태)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    for s in batch.next_state:
        if s is not None:
            non_final_next_states = torch.stack(tuple(torch.Tensor(s)))

    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
    # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
    state_action_values = policy_net(state_batch).gather(-1, action_batch.squeeze(1))

    # 모든 다음 상태를 위한 V(s_{t+1}) 계산
    # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
    # max(1)[0]으로 최고의 보상을 선택하십시오.
    # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(-1)[0].detach()
    # 기대 Q 값 계산
    expected_state_action_values = (next_state_values * DISCOUNT_FACTOR) + reward_batch

    # Huber 손실 계산
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    losses.append(loss.item())

    # 모델 최적화
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


######################################################################
#
# 아래에서 주요 학습 루프를 찾을 수 있습니다. 처음으로 환경을
# 재설정하고 ``상태`` Tensor를 초기화합니다. 그런 다음 행동을
# 샘플링하고, 그것을 실행하고, 다음 화면과 보상(항상 1)을 관찰하고,
# 모델을 한 번 최적화합니다. 에피소드가 끝나면 (모델이 실패)
# 루프를 다시 시작합니다.
#
# 아래에서 `num_episodes` 는 작게 설정됩니다. 노트북을 다운받고
# 의미있는 개선을 위해서 300 이상의 더 많은 에피소드를 실행해 보십시오.
#
throughputs = []
rewards = []
scatters_front = []
scatters_middle = []
scatters_tail = []

show_state = []
show_next_states = []
i_episode = 0
reward_count = 0
maxs=[]

for i_episode in tqdm(range(num_episodes)):
    # 환경과 상태 초기화
    state = env.reset()
    state = torch.Tensor(state)
    throughput_count=0
    # print('state type : ',state.type())
    max_count=0
    for t in count():
        # 행동 선택과 수행
        action = select_action(state)
        next_state, reward, done, last_set = env.step(action.item())

        state_reshape = np.reshape(state, (env.N+2, 4))
        next_state_reshape = np.reshape(next_state, (env.N+2, 4))

        if i_episode > (num_episodes - 3):
            print(state_reshape[0], next_state_reshape[0],
                  "action:%2d%2d%2d%2d throughput:%6.3f foot:%6.3f dispersed:%6.3f move:%6.3f txr:%3d"
                  % (last_set[5], last_set[6], last_set[7], last_set[8],
                     last_set[0], last_set[1], last_set[2], last_set[3], last_set[4]))

        if last_set[0] != 0:
            throughput_count += 1

        rewards.append(reward)
        reward = torch.tensor([reward], device=device)

        if i_episode == (num_episodes - 1):
            scatters_tail.append(np.array(next_state_reshape[0]))
        elif i_episode == num_episodes // 2:
            scatters_middle.append(np.array(next_state_reshape[0]))
        elif i_episode == 0:
            scatters_front.append(np.array(next_state_reshape[0]))

        next_state = torch.Tensor(next_state)
        if done:
            next_state = None

        # 메모리에 변이 저장
        memory.push(state, action, next_state, reward)

        if torch.argmax(policy_net(state)) == action:
            max_count += 1

        # 다음 상태로 이동
        state = next_state

        # (정책 네트워크에서) 최적화 한단계 수행
        optimize_model()
        if done:
            # episode_durations.append(t + 1)
            # plot_durations()
            break

    maxs.append(max_count)
    # 목표 네트워크 업데이트, 모든 웨이트와 바이어스 복사
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    throughputs.append(throughput_count)

f=open(path+'/'+str,'w')
torch.save({
    'target_net': target_net.state_dict(),
    'policy_net': policy_net.state_dict(),
    'optimizer': optimizer.state_dict()
}, path+'/'+str)

print('Complete')

print(losses.__len__())
def get_mean(array):
    means = []
    for n in range(0,num_episodes,1):
        sum = 0
        for i in range(0,env.MAX_STEPS,1):
            sum += array[(n*env.MAX_STEPS)+i]
        means.append(sum / env.MAX_STEPS)
    return means

def get_reward_mean2(array):
    means = []
    for n in range(0,num_episodes//50,1):
        sum = 0
        for i in range(50):
            sum+=array[n*(50)+i]
        means.append(sum/50)
    return means

'''plt.figure()
plt.title('max_count')
plt.xlabel('episode')
plt.ylabel('count')
plt.plot(maxs)'''

plt.figure()
plt.title('throughput')
plt.xlabel('episode')
plt.ylabel('throughput_count')
plt.plot(throughputs)

"""plt.figure()
plt.title('throughput')
plt.xlabel('step')
plt.ylabel('throughput')
x_values = list(range(throughputs.__len__()))
y_values = [y for y in throughputs]
plt.scatter(x_values, y_values, s=40)"""

'''plt.figure()
plt.title('reward')
plt.xlabel('step')
plt.ylabel('Reward')
plt.plot(rewards)'''

reward_means = get_mean(rewards)
plt.figure()
plt.title('reward mean')
plt.xlabel('episode')
plt.ylabel('Reward')
plt.plot(reward_means)

reward_means2 = get_reward_mean2(reward_means)
plt.figure()
plt.title('reward mean2')
plt.xlabel('50 episodes')
plt.ylabel('Reward')
plt.plot(reward_means2)

plt.figure()
plt.title('eps')
plt.xlabel('step')
plt.ylabel('epsilon')
plt.plot(epslions)

loss_means = get_mean(losses)
plt.figure()
plt.title('loss')
plt.xlabel('episode')
plt.ylabel('loss')
plt.plot(loss_means)

"""def create_sphere(cx, cy, cz, r):
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    # shift and scale sphere
    x = r * x + cx
    y = r * y + cy
    z = r * z + cz
    return (x, y, z)"""

# 3D 그래프 그리기
'''fig = plt.figure()
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
"""for i in range(0, env.MAX_STEPS//10, 1):
    for j in range(10):
        # 구 그리기
        (x, y, z) = create_sphere(scatters[i][0], scatters[i][1], scatters[i][2], scatters[i][3])
        ax.auto_scale_xyz([0, 500], [0, 500], [0, 0.15])
        ax.plot_surface(x, y, z, color=color_list[i%6], linewidth=0, alpha=0.1)
        # 점 찍기
        ax.scatter(np.transpose(scatters)[0], np.transpose(scatters)[1], np.transpose(scatters)[2],marker='o', s=60, c=color_list[i])"""
ax.scatter(np.transpose(scatters_front)[0], np.transpose(scatters_front)[1], np.transpose(scatters_front)[2],
           marker='o', s=60, c='orange')
ax.scatter(np.transpose(scatters_middle)[0], np.transpose(scatters_middle)[1], np.transpose(scatters_middle)[2],
           marker='o', s=60, c='red')
ax.scatter(np.transpose(scatters_tail)[0], np.transpose(scatters_tail)[1], np.transpose(scatters_tail)[2], marker='o',
           s=60, c='purple')

ax.scatter(np.transpose(nodes)[0], np.transpose(nodes)[1], np.transpose(nodes)[2], marker='o', s=80, c='cyan')
plt.show()'''
# env.render()
env.close()
plt.ioff()
plt.show()

######################################################################
# 다음은 전체 결과 데이터 흐름을 보여주는 다이어그램입니다.
#
# .. figure:: /_static/img/reinforcement_learning_diagram.jpg
#
# 행동은 무작위 또는 정책에 따라 선택되어, gym 환경에서 다음 단계 샘플을 가져옵니다.
# 결과를 재현 메모리에 저장하고 모든 반복에서 최적화 단계를 실행합니다.
# 최적화는 재현 메모리에서 무작위 배치를 선택하여 새 정책을 학습합니다.
# "이전" target_net은 최적화에서 기대 Q 값을 계산하는 데에도 사용되고,
# 최신 상태를 유지하기 위해 가끔 업데이트됩니다.
#
