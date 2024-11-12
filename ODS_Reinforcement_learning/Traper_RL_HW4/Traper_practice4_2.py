import torch
import torch.nn as nn
import gym
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

env = gym.make('CartPole-v1')
state_n = 4
action_n = 2

def discretize_state(state, levels=10):
    if isinstance(state, (np.ndarray, list)):
        # Если `state` — массив, дискретизируем каждый элемент
        discrete_state = np.round(np.array(state) * levels) / levels
        return tuple(discrete_state)  # Преобразуем в кортеж для использования в Q-таблице
    else:
        # Если `state` — одиночное число, просто дискретизируем его
        return round(state * levels) / levels

def get_epsilon_greedy_action(q_values, epsilon, action_n):
    prob = np.ones(action_n) * epsilon / action_n
    argmax_action = np.argmax(q_values)
    prob[argmax_action] += 1 - epsilon
    action = np.random.choice(np.arange(action_n), p=prob)
    return action

def MonteCarlo(state_n, action_n, env, episode_n, trajectory_len=500, gamma=0.99):
    q_values = defaultdict(lambda: np.zeros(action_n))
    #q_values = defaultdict(lambda: np.ones(action_n) * 10)  # Начальные значения Q-значений больше 0

    counters = defaultdict(lambda: np.zeros(action_n))

    levels = 10

    total_rewards = []
    for episode in range(episode_n):
        print(f'Episode = {episode}')
        epsilon = 1 - episode / episode_n
        #epsilon = max(0.01, 1 - episode / (episode_n * 3))  # Более медленное уменьшение ε
        #epsilon = max(0.1, 1 - episode / (episode_n * 1.5))  # Начальное значение ε ограничено минимумом в 0.1
        #gamma = np.random.uniform(0.9, 0.99)

        trajectory = {'states': [], 'actions': [], 'rewards': []}

        state = env.reset()
        state = tuple([discretize_state(st, levels=levels) for st in state])

        for t in range(trajectory_len):
            action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)
            next_state, reward, done, _ = env.step(action)
            next_state = tuple([discretize_state(st, levels=levels) for st in next_state])

            trajectory['states'].append(state)
            trajectory['actions'].append(action)
            trajectory['rewards'].append(reward)

            state = next_state
            
            if done:
                break

        total_rewards.append(np.sum(trajectory['rewards']))
        
        real_trajectory_len = len(trajectory['rewards'])
        returns = np.zeros(real_trajectory_len + 1)
        for t in range(real_trajectory_len - 1, -1, -1):
            returns[t] = trajectory['rewards'][t] + gamma * returns[t + 1]

        for t in range(real_trajectory_len):
            state = trajectory['states'][t]
            action = trajectory['actions'][t]
            counters[state][action] += 1
            q_values[state][action] += (returns[t] - q_values[state][action]) / counters[state][action]

        print(total_rewards[-1])

    return total_rewards

def SARSA(action_n, state_n, env, episode_n, gamma=0.99, trajectory_len=500, alpha=0.5, levels=10):
    q_values = defaultdict(lambda: np.zeros(action_n))

    total_rewards = []
    for episode in range(episode_n):
        epsilon = 1 - episode / episode_n

        total_reward = 0

        state = env.reset()
        state = tuple([discretize_state(st, levels=levels) for st in state])
        action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)
        for t in range(trajectory_len):
            next_state, reward, done, _ = env.step(action)
            next_state = tuple([discretize_state(st, levels=levels) for st in next_state])
            
            next_action = get_epsilon_greedy_action(q_values[next_state], epsilon, action_n)

            q_values[state][action] += alpha * (reward + gamma * q_values[next_state][next_action] - q_values[state][action])

            total_reward += reward

            state = next_state
            action = next_action

            if done:
                break

        total_rewards.append(total_reward)
    
    return total_rewards

def QLearning(action_n, state_n, env, episode_n, noisy_episode_n, gamma=0.99, t_max=500, alpha=0.5, levels=10):
    q_values = defaultdict(lambda: np.zeros(action_n))

    total_rewards = []

    for episode in range(episode_n):
        epsilon = 1 - episode / episode_n

        total_reward = 0

        state = env.reset()
        state = tuple([discretize_state(st, levels=levels) for st in state])
        action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)
        
        for t in range(t_max):
            next_state, reward, done, _ = env.step(action)
            next_state = tuple([discretize_state(st, levels=levels) for st in next_state])
            next_action = get_epsilon_greedy_action(q_values[next_state], epsilon, action_n)

            q_values[state][action] += alpha * (reward + gamma * np.max(q_values[next_state]) - q_values[state][action])

            total_reward += reward

            state = next_state
            action = next_action

            if done:
                break
        #print(total_reward)
        total_rewards.append(total_reward)
    
    return total_rewards

class DCEM(nn.Module):
    def __init__(self, state_dim, action_n, epsilon=0, epsilon_decrease=1e-3, epsilon_min=1e-2, hidden_size=128, lr=1e-2):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n
        self.network = nn.Sequential(nn.Linear(self.state_dim, hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(hidden_size, self.action_n)
                                    )
        self.softmax = nn.Softmax()
        self.optimazer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()
        self.epsilon = epsilon
        self.epsilon_descrease = epsilon_decrease
        self.epsilon_min = epsilon_min

    def forward(self, _input):
        return self.network(_input)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.forward(state)
        #probs = self.softmax(logits)
        probs = (1 - self.epsilon) * self.softmax(logits) + self.epsilon / self.action_n
        action = np.random.choice(self.action_n, p=probs.data.numpy())
        return action

    def fit(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                elite_states.append(state)
                elite_actions.append(action)

        # Преобразуем списки в numpy.ndarray для повышения эффективности
        elite_states = np.array(elite_states)
        elite_actions = np.array(elite_actions)

        elite_states = torch.FloatTensor(elite_states)
        elite_actions = torch.LongTensor(elite_actions)
        pred_actions = self.forward(elite_states)

        loss = self.loss(pred_actions, elite_actions)
        loss.backward()
        self.optimazer.step()
        self.optimazer.zero_grad()

        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_descrease)

def get_trajectory(env, agent, max_len=1000):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    state = env.reset()

    for _ in range(max_len):
        state = discretize_state(state, 10)
        trajectory['states'].append(state)

        action = agent.get_action(state)
        trajectory['actions'].append(action)

        state, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)

        if done:
            break

    return trajectory

def DCEM_fit_agent(env, state_dim, action_n, iteration_n, epsilon, hidden_size, lr, trajectory_n, trajectory_len, q):
    agent = DCEM(state_dim, action_n, epsilon_decrease=1/iteration_n, epsilon=epsilon, hidden_size=hidden_size, lr=lr)
    mean_total_rewards = []
    for iteration in range(iteration_n):
        #policy evaluation
        trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
        total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
        print('iteration:', iteration, 'mean total reward:', np.mean(total_rewards))

        #mean_total_rewards.append(np.mean(total_rewards))
        mean_total_rewards.extend([np.mean(total_rewards)] * trajectory_n)

        #policy improvement
        quantile = np.quantile(total_rewards, q)
        elite_trajectories = []
        for trajectory in trajectories:
            total_reward = np.sum(trajectory['rewards'])
            if total_reward > quantile:
                elite_trajectories.append(trajectory)

        if len(elite_trajectories) > 0:
            agent.fit(elite_trajectories)

    trajectory = get_trajectory(env, agent, max_len=trajectory_len)
    return mean_total_rewards

def moving_average(data, window):
    return np.convolve(data, np.ones(window) / window, mode='valid')

window = 100

def plot_results(QLearning_total_rewards, SARSA_total_rewards, MonteCarlo_total_rewards, CrossEntropy_total_rewards):
    QLearning_trajectories = np.arange(1, len(QLearning_total_rewards) + 1)
    plt.plot(QLearning_trajectories[:len(moving_average(QLearning_total_rewards, window))], moving_average(QLearning_total_rewards, window), label='Q-Learning')

    SARSA_trajectories = np.arange(1, len(SARSA_total_rewards) + 1)
    plt.plot(SARSA_trajectories[:len(moving_average(SARSA_total_rewards, window))], moving_average(SARSA_total_rewards, window), label='SARSA')

    MonteCarlo_trajectories = np.arange(1, len(MonteCarlo_total_rewards) + 1)
    plt.plot(MonteCarlo_trajectories[:len(moving_average(MonteCarlo_total_rewards, window))], moving_average(MonteCarlo_total_rewards, window), label='MonteCarlo')

    CrossEntropy_trajectories = np.arange(1, len(CrossEntropy_total_rewards) + 1)
    plt.plot(CrossEntropy_trajectories[:len(moving_average(CrossEntropy_total_rewards, window))], moving_average(CrossEntropy_total_rewards, 10), label='CrossEnthropy')

    plt.legend()
    plt.grid()
    plt.show()

CrossEntropy_total_rewards = DCEM_fit_agent(iteration_n=50, epsilon=0.3, hidden_size=128, lr=1e-2, trajectory_n=50, trajectory_len=300, q=0.8)
QLearning_total_rewards = QLearning(env, episode_n=20000, noisy_episode_n=400, t_max=500, gamma=0.999, alpha=0.5, levels=10)
SARSA_total_rewards = SARSA(env, episode_n=20000, trajectory_len=500, gamma=0.999, alpha=0.3, levels=10)
MonteCarlo_total_rewards = MonteCarlo(env, episode_n=20000, trajectory_len=500, gamma=0.99, levels=10)

plot_results(QLearning_total_rewards, SARSA_total_rewards, MonteCarlo_total_rewards, CrossEntropy_total_rewards)