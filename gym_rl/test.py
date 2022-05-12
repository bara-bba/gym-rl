import gym
import gym_rl
from gym_rl.envs import PandaEnv

def run_one_episode (env):
    env.reset()
    sum_reward = 0
    for i in range(20):
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        sum_reward += reward
        if done:
            break
    return sum_reward

env = gym.make("PandaEnv-v0")
history = []
for _ in range(1000):
    sum_reward = run_one_episode(env)
    history.append(sum_reward)
avg_sum_reward = sum(history) / len(history)
print("\nbaseline cumulative reward: {:6.2}".format(avg_sum_reward))