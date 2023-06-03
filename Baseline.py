import random
import matplotlib.pyplot as plt
from tempfile import TemporaryFile
import gymnasium as gym
import numpy as np


env = gym.make("ALE/CrazyClimber-v5", render_mode="rgb_array")


def is_record(ep):
    if ep == 0 or ep == 198:
        return True
    return False


env = gym.wrappers.RecordVideo(env, "./recordings", episode_trigger=is_record)

rewards = []
num_episodes = 200
for i in range(num_episodes):
    observation, _ = env.reset()
    total_reward = 0
    timesteps = 0
    reward = 0

    done = False
    while not done:
        action = random.randint(1, 8)  # User-defined policy function
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        timesteps += 1
        total_reward += reward

    rewards.append(total_reward)
    print('Completed Episode: ', i + 1, ' with return: ', total_reward, ' at timestep: ',
          timesteps)
env.close()

episode_list = []
for i in range(num_episodes):
    episode_list.append(i+1)

plt.plot(episode_list,rewards)
plt.title('Performance of the baseline algorithm on Crazy Climbers')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.savefig('baseline.png', bbox_inches='tight')

with open("baseline_results.txt", 'a') as f:
    f.write(str(rewards) + "\n")

