import gym
import gym_rl
from gym_rl.envs import PandaEnv

def run_one_episode (env):
    env.reset()
    sum_reward = 0
    for i in range(env.MAX_STEPS):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        sum_reward += reward
        if done:
            break
    return sum_reward

env = gym.make("PandaEnv-v0")
sum_reward = run_one_episode(env)
