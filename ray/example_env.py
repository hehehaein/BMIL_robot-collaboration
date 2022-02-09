import gym
import numpy as np
from gym.utils import seeding
import random

# ~number of relay node
N = 2
# ~transmission radius max
R_MAX = 4
#location x,y,z
MIN_LOC = 0
MAX_LOC = 4

class Example_v0 (gym.Env):


    # possible actions
    MOVE_LF = 0
    MOVE_RT = 1

    # possible positions
    LF_MIN = 1
    RT_MAX = 10

    # land on the GOAL position within MAX_STEPS steps
    MAX_STEPS = 10

    # possible rewards
    REWARD_AWAY = -2
    REWARD_STEP = -1
    REWARD_GOAL = MAX_STEPS

    metadata = {
        "render.modes": ["human"]
        }


    """def __init__ (self):
        # ~the action space ranges [x,y,z,r] where {-1,0,1}:
        # ~0 :-1
        # ~1 :0
        # ~2 :1
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 3, 3])

        # NB: Ray throws exceptions for any `0` value Discrete
        # observations so we'll make position a 1's based value
        self.observation_space = gym.spaces.Box(0, R_MAX, shape=(N+2, 4), dtype=np.int8)  # origin : self.observation_space = ~~

        # possible positions to chose on `reset()`
        # self.goal = int((self.LF_MIN + self.RT_MAX - 1) / 2)

        self.init_positions = list(range(self.LF_MIN, self.RT_MAX))
        self.init_positions.remove(self.goal)

        # NB: change to guarantee the sequence of pseudorandom numbers
        # (e.g., for debugging)
        self.seed()

        self.reset()"""

    def __init__(self,i) :
        #self.state_space = np.empty((N + 2, 4), int)
        self.action_space = np.arange(-1,1,1) #(-1,0,1)
        self.agent_i = i

        """for i in range(N+2):
            for j in range(4):
                k = np.random.randint(MIN_LOC, MAX_LOC) #randint : 균일분포의 정수 난수 1개
                self.state_space[i][j]=k

        for i in range(3):
            k = np.random.randint(MIN_LOC, MAX_LOC)
            self.action_space[i]=k
        k=np.random.randint(0,R_MAX)
        self.action_space[3]=k  # txr 설정"""




    def reset (self, source, destination, max_height):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """



        #self.position = self.np_random.choice(self.init_positions)
        self.count = 0

        self.state = np.empty((N + 2, 4), int)
        #for i in range (0,N,1) :
        #    self.state_array[i+2,:]= [destination[0], destination[1], max_height, R_MAX]
        #self.state[N, :] = source[:] #source
        #self.state[N+1, :] = destination[:] #destination

        self.state[0, :] = [destination[0], destination[1], max_height, R_MAX]
        self.state[1, :] = [3, 3, 3, 3]
        self.state[2, :] = source[:] #source
        self.state[3, :] = destination[:] #destination

        self.reward = 0
        self.done = False
        self.info = {}

        return self.state


    def check_state(self,state):
        #노드의 위치가 이동구간을 벗어갔을때 return 1
        for i in range(N+2):
            for j in range(3):
                if state[i][j] < MIN_LOC or MAX_LOC < state[i][j] :
                    return 1

        # 노드의 txr이 범위를 벗어났을때 return 1
        for i in range(N + 2):
            if state[i][3] < 0 or R_MAX < state[i][3] :
                return 1

        #노드의 위치가 이동구간 안에 있고, txr도 범위안에 있으면 return 0
        return 0

        return 0

    def step (self, action):
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
            self.done = True;

        else:
            assert self.action_space.contains(action)
            self.count += 1



            self.info["dist"] = self.goal - self.position

        try:
            assert self.observation_space.contains(self.state)
        except AssertionError:
            print("INVALID STATE", self.state)

        return [self.state, self.reward, self.done, self.info]


    def render (self, mode="human"):
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
        s = "position: {:2d}  reward: {:2d}  info: {}"
        print(s.format(self.state, self.reward, self.info))


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