import numpy as np
import random
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make("ALE/CrazyClimber-v5", render_mode="rgb_array")


def is_record(ep):
    if ep == 0 or ep == 198:
        return True
    return False


env = gym.wrappers.RecordVideo(env, "./recordings", episode_trigger=is_record)


class SARSA:
    def __init__(self, alpha, gamma, epsilon):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def parse_state_totuple(self, state):
        state_list = state.tolist()
        state = []
        for matrix in state_list:
            for row_list in matrix:
                tuple_list = tuple(row_list)
                state.append(tuple_list)
        return tuple(state)

    def update_state(self, state):
        if state not in self.q_table:
            self.q_table[state] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

    def get_best_action(self, state):
        q = self.q_table[state][0]
        a = 0
        for action, q_value in self.q_table[state].items():
            if q_value > q:
                q = q_value
                a = action
        return a, q

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = random.randint(0, 8)
        else:
            action, q_value = self.get_best_action(state)
        return action

    def learn(self, state, action, reward, next_state, next_action, done):
        td_error = reward + self.gamma * self.q_table[next_state][next_action] - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def train(self, env, num_episodes):
        reward_list = []
        for i in range(num_episodes):
            print("Episode ", i+1)
            reward_sum = 0

            state, _ = env.reset()
            # print(state)
            state = self.parse_state_totuple(state)
            self.update_state(state)
            action = self.choose_action(state)

            # Episodic loop
            done = False
            while not done:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                next_state = self.parse_state_totuple(next_state)
                self.update_state(next_state)

                reward_sum += reward
                next_action = self.choose_action(next_state)
                self.learn(state, action, reward, next_state, next_action, done)

                state = next_state
                action = next_action

            print("Completed. Reward : ", reward_sum)
            reward_list.append(reward_sum)
        return reward_list


# hyperparameters
alpha = 0.3
epsilon = 0.1
gamma = 0.8
num_actions = 9
num_episodes = 200
modified_agent_rewards = []

# initialise agent
agent = SARSA(alpha, epsilon, gamma)

# train agent
agent_reward = agent.train(env, num_episodes)
modified_agent_rewards.append(agent_reward)

env.close()

# produce graph
episode_list = []
for i in range(num_episodes):
    episode_list.append(i + 1)

plt.plot(episode_list, agent_reward)
plt.title('Performance of the SARSA on Crazy Climbers')
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.savefig('sarsa.png', bbox_inches='tight')

with open("sarsa2_results.txt", 'a') as f:
    f.write(str(agent_reward) + "\n")
