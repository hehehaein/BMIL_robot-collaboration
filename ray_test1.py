import ray 
from ray.tune.registry import register_env 
from ray.rllib.agents import ppo 
from ray import tune

ray.init()
tune.run("DQN",
         stop={"episode_reward_mean":100,
               "timesteps_total": 100000},
         config={'env':'CartPole-v0',
                 "time"})
cartpole-dqn:
    env: CartPole-v0
    run: DQN
    stop:
        episode_reward_mean: 100
        timesteps_total: 100000
    config:
        # Works for both torch and tf.
        framework: tf
        model:
            fcnet_hiddens: [64]
            fcnet_activation: linear
        n_step: 3