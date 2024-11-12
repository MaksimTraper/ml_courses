import numpy as np
import gym
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3")

state_n, action_n = env.observation_space.n, env.action_space.n

def get_epsilon_greedy_action(q_values, epsilon, action_n):
    prob = np.ones(action_n) * epsilon / action_n
    argmax_action = np.argmax(q_values)
    prob[argmax_action] += 1 - epsilon
    action = np.random.choice(np.arange(action_n), p=prob)
    return action

def MonteCarlo(env, episode_n, trajectory_len=500, gamma=0.99):
    state_n = env.observation_space.n 
    action_n = env.action_space.n

    q_values = np.zeros((state_n, action_n))
    counters = np.zeros((state_n, action_n))

    total_rewards = []
    for episode in range(episode_n):
        epsilon = 1 - episode / episode_n

        trajectory = {'states': [], 'actions': [], 'rewards': []}

        state = env.reset()
        for t in range(trajectory_len):
            action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)
            next_state, reward, done, _ = env.step(action)

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

    return total_rewards

def SARSA(env, episode_n, gamma=0.99, trajectory_len=500, alpha=0.5):
    state_n = env.observation_space.n 
    action_n = env.action_space.n

    q_values = np.zeros((state_n, action_n))

    total_rewards = []
    for episode in range(episode_n):
        epsilon = 1 - episode / episode_n

        total_reward = 0

        state = env.reset()
        action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)
        for t in range(trajectory_len):
            next_state, reward, done, _ = env.step(action)
            next_action = get_epsilon_greedy_action(q_values[next_state], epsilon, action_n)

            q_values[state][action] += alpha * (reward + gamma * q_values[next_state][next_action] - q_values[state][action])

            total_reward += reward

            state = next_state
            action = next_action

            if done:
                break

        total_rewards.append(total_reward)
    
    return total_rewards

def QLearning(env, episode_n, noisy_episode_n, gamma=0.99, t_max=500, alpha=0.5):
    state_n = env.observation_space.n 
    action_n = env.action_space.n

    q_values = np.zeros((state_n, action_n))

    total_rewards = []

    for episode in range(episode_n):
        epsilon = 1 - episode / episode_n

        total_reward = 0

        state = env.reset()
        action = get_epsilon_greedy_action(q_values[state], epsilon, action_n)
        
        for t in range(t_max):
            next_state, reward, done, _ = env.step(action)
            next_action = get_epsilon_greedy_action(q_values[next_state], epsilon, action_n)

            q_values[state][action] += alpha * (reward + gamma * np.max(q_values[next_state]) - q_values[state][action])

            total_reward += reward

            state = next_state
            action = next_action

            if done:
                break
        
        total_rewards.append(total_reward)
    
    return total_rewards

class CrossEntropyAgent():
    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n

        self.model = np.ones((self.state_n, self.action_n)) / self.action_n

    def get_action(self, state):
        probs = self.model[state]
        normalized_probs = probs / np.sum(probs)  # нормализация вероятностей
        action = np.random.choice(np.arange(self.action_n), p=normalized_probs)
        return int(action)

    def fit(self, elite_trajectories, smoothing, coef_lambda):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                if smoothing == 'Policy':
                    if(0 < coef_lambda <= 1):
                        new_model[state] = coef_lambda * new_model[state] + (1 - coef_lambda) * self.model[state].copy()
                    else:
                        print('Выбран неправильный коэффициент лямбда для сглаживания политики')
                        return
            else:
                new_model[state] = self.model[state].copy()
        self.model = new_model
        return None

def get_state(obs):
    return int(np.sqrt(state_n) * obs[0] + obs[1])

def get_trajectory(env, agent, trajectory_len):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    obs = env.reset()
    state = obs

    for _ in range(trajectory_len):
        trajectory['states'].append(state)

        action = agent.get_action(state)
        trajectory['actions'].append(action)
        
        obs, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)
        
        state = obs

        if done:
            break
    
    return trajectory

def fit_agent(q_param, iteration_n, trajectory_n, smoothing, coef_lambda):
    agent = CrossEntropyAgent(state_n, action_n)
    mean_total_rewards = []

    for iteration in range(iteration_n):

        #policy evaluation
        trajectories = [get_trajectory(env, agent, 1000) for _ in range(trajectory_n)]
        total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
        
        #mean_total_rewards.append(np.mean(total_rewards))
        mean_total_rewards.extend([np.mean(total_rewards)] * trajectory_n)
        #print(f'Iteration {iteration}, mean_total_reward = {mean_total_rewards[-1]}')

        new_q_param = q_param

        #policy improvement
        quantile = np.quantile(total_rewards, new_q_param)
        elite_trajectories = []
        for trajectory, total_reward in zip(trajectories, total_rewards):
            if total_reward > quantile:
                elite_trajectories.append(trajectory)

        agent.fit(elite_trajectories, smoothing, coef_lambda)

    trajectory = get_trajectory(env, agent, trajectory_len=100)

    return mean_total_rewards

def moving_average(data, window):
    return np.convolve(data, np.ones(window) / window, mode='valid')

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

CrossEntropy_total_rewards = fit_agent(q_param=0.4, iteration_n=16, trajectory_n=400, smoothing='Policy', coef_lambda=0.95)
QLearning_total_rewards = QLearning(env, episode_n=500, noisy_episode_n=400, t_max=1000, gamma=0.999, alpha=0.5)
SARSA_total_rewards = SARSA(env, episode_n=2000, trajectory_len=1000, gamma=0.999, alpha=0.5)
MonteCarlo_total_rewards = MonteCarlo(env, episode_n=2000, trajectory_len=1000, gamma=0.99)

plot_results(QLearning_total_rewards, SARSA_total_rewards, MonteCarlo_total_rewards, CrossEntropy_total_rewards)