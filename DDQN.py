import random
import gymnasium as gym
import numpy as np
import math

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from collections import deque

import matplotlib.pyplot as plt
import gc

# environment and preprocessing wrappers
env = gym.make("ALE/CrazyClimber-v5", render_mode='rgb_array')
env = gym.wrappers.GrayScaleObservation(env, keep_dim=False)
env = gym.wrappers.ResizeObservation(env, (84, 84))


def is_record(ep):
    if ep == 0 or ep == 198:
        return True
    return False


env = gym.wrappers.RecordVideo(env, "./recordings", episode_trigger=is_record)

# hyperparameters
gamma = 0.99
eps = 0.99
eps_min = 0.05
eps_decay = 0.999994
learn_rate = 0.00025
batch_size = 32

num_episodes = 200
max_timesteps = 50000
buffer_size = 100000
update_target = 1000


# replay buffer and sampling
class Buffer:
    def __init__(self):
        self.memory = deque(maxlen=buffer_size)

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_batch(self, batch):
        samples = random.sample(self.memory, min(batch, len(self.memory)))
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for sample in samples:
            states.append(sample[0])
            actions.append(sample[1])
            rewards.append(sample[2])
            next_states.append(sample[3])
            dones.append(sample[4])

        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)


class Agent:
    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.state_space = self.env.observation_space.shape[0]

        # main and target networks
        self.model = self.build_network()
        self.target_model = self.build_network()

    # network model
    def build_network(self):
        model = Sequential([
            Conv2D(32, (8, 8), strides=4, padding='same', input_shape=(84, 84, 1)),
            Activation('relu'),
            Conv2D(64, (4, 4), strides=2, padding='same'),
            Activation('relu'),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(self.action_size, activation='linear', dtype=tf.float32)
        ])
        model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(learning_rate=learn_rate))
        return model

    # updates target network with main network weights
    def update(self):
        self.target_model.set_weights(self.model.get_weights())

    # picks random/best action
    def act(self, in_state, eps, frames):
        eps_threshold = 1
        # delayed epsilon decay
        if frames > 18000:
            eps_threshold = eps_min + (eps - eps_min) * math.exp(-1. * (timestep - 18000) * (1 - eps_decay))
        epsilons.append(eps_threshold)
        if np.random.random() < eps_threshold:
            return np.random.randint(0, self.action_size)
        in_state = tf.convert_to_tensor(in_state[None, :], dtype=tf.float32)
        action_qval = self.model.predict_on_batch(in_state)
        best_action = np.argmax(action_qval[0], axis=0)
        return best_action

    # training function
    def train(self, batch):
        # calculating q values
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
        curr_q = self.model.predict_on_batch(state_batch)
        target_q = np.copy(curr_q)
        max_next_q = self.target_model.predict_on_batch(next_state_batch)[
            range(batch_size), np.argmax(self.model.predict_on_batch(next_state_batch), axis=1)]
        # updating q values
        for i in range(state_batch.shape[0]):
            target_q_val = reward_batch[i]
            if not done_batch[i]:
                target_q_val += gamma * max_next_q[i]
            target_q[i][action_batch[i]] = target_q_val
        training_history = self.model.fit(x=state_batch, y=target_q, verbose=0)


# creating agent and buffer
agent = Agent(env)
buffer = Buffer()

rewards = []
epsilons = []
losses = []
avg_rewards = []

# main loop
timestep = 0
for i in range(num_episodes):
    print('Episode: ', i + 1)
    state, _ = env.reset()
    total_reward = 0
    loss_temp = 0
    floor = 0
    prev_life = 5
    finish_time = max_timesteps

    # episodic loop
    done = False
    while not done:
        timestep += 1
        action = agent.act(state, eps, timestep)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # storing in replay buffer
        buffer.store(state, action, reward, next_state, done)

        # training every 2 timesteps
        if batch_size < len(buffer.memory) and timestep % 2 == 0:
            # print("training!")
            sample = buffer.sample_batch(batch_size)
            agent.train(sample)
        state = next_state

        # updating target network
        if timestep % update_target == 0:
            # print("agent update at ", timestep)
            agent.update()
        if done:
            finish_time = timestep

    # outputting episode results
    rewards.append(total_reward)
    print('Episode: ', i + 1, ' with return: ', total_reward, ' at timestep: ',
          finish_time)

env.close()


# produce graph
episode_list = []
for i in range(num_episodes):
    episode_list.append(i + 1)

plt.plot(episode_list, rewards)
plt.title('Performance of Double DQN on Crazy Climbers')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.savefig('DDQN.png', bbox_inches='tight')

with open("DDQN_results.txt", 'a') as f:
    f.write(str(rewards) + "\n")
