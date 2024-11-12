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

def MonteCarlo(env, state_n, action_n, episode_n, epsilon_fun, trajectory_len=500, gamma=0.99):
    q_values = np.zeros((state_n, action_n))
    counters = np.zeros((state_n, action_n))

    total_rewards = []
    for episode in range(1, episode_n):
        last_reward = 0
        if len(total_rewards) > 0:
            last_reward = total_rewards[-1]

        epsilon = epsilon_fun(episode, episode_n, last_reward=last_reward, total_rewards=total_rewards)

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

def moving_average(data, window):
    return np.convolve(data, np.ones(window) / window, mode='valid')

window = 10
def plot_results(MonteCarloDict):

    for label, MonteCarlo_total_rewards in MonteCarloDict.items():
        MonteCarlo_trajectories = np.arange(1, len(MonteCarlo_total_rewards) + 1)
        plt.plot(MonteCarlo_trajectories[:len(moving_average(MonteCarlo_total_rewards, window))], moving_average(MonteCarlo_total_rewards, window), label=label)

    plt.legend()
    plt.grid()
    plt.title('Сравнение различных алгоритмов выбора эпсилон в методе Monte-Carlo')
    plt.xlabel('№ траектории')
    plt.ylabel('Награда')
    plt.show()

dict = {}

def calculate_epsilon(episode, episode_n, last_reward):
    epsilon = 1 - episode / episode_n
    return epsilon

dict['1 - episode/episode_n'] = (MonteCarlo(env, epsilon_fun=calculate_epsilon, state_n=state_n, action_n=action_n, episode_n=2000, trajectory_len=1000, gamma=0.99))

def calculate_epsilon(episode, episode_n, last_reward):
    epsilon = 1 / episode
    return epsilon

dict['1/k'] = (MonteCarlo(env, state_n=state_n, action_n=action_n, epsilon_fun=calculate_epsilon, episode_n=2000, trajectory_len=1000, gamma=0.99))

def calculate_epsilon(episode, episode_n, last_reward):
    epsilon = max(0.1, 1 - episode / (episode_n * 1.5))
    return epsilon

dict['max(0.1, 1 - episode / (episode_n * 1.5))'] = (MonteCarlo(env, state_n=state_n, action_n=action_n, epsilon_fun=calculate_epsilon, episode_n=2000, trajectory_len=1000, gamma=0.99))

def calculate_epsilon(episode, episode_n, last_reward):
    epsilon = (episode_n - episode) / episode_n
    return epsilon

dict['(episode_n - episode) / episode_n'] = (MonteCarlo(env, state_n=state_n, action_n=action_n, epsilon_fun=calculate_epsilon, episode_n=2000, trajectory_len=1000, gamma=0.99))

def calculate_epsilon(episode, episode_n, last_reward):
    epsilon = 0.1 + (1 - 0.1) * np.e**(-0.01*episode)
    return epsilon

dict['0.1 + (1 - 0.1) * np.e**(-0.01*episode)'] = (MonteCarlo(env, state_n=state_n, action_n=action_n, epsilon_fun=calculate_epsilon, episode_n=2000, trajectory_len=1000, gamma=0.99))

def calculate_epsilon(episode, episode_n, last_reward):
    epsilon = min(0.1, 1 - 2*last_reward/500)
    return epsilon

dict['min(0.1, 1 - 2*last_reward/500)'] = (MonteCarlo(env, state_n=state_n, action_n=action_n, epsilon_fun=calculate_epsilon, episode_n=2000, trajectory_len=1000, gamma=0.99))

def calculate_epsilon(episode, episode_n, last_reward):
    epsilon_decay = 1 - episode / episode_n  # уменьшение с каждым эпизодом
    epsilon_reward_based = 1 - 2 * last_reward / 500
    epsilon = min(0.1, max(epsilon_decay, epsilon_reward_based))
    return epsilon

dict['min(0.1, 1 - max(1 - episode / episode_n, 1 - 2 * last_reward / 500)'] = (MonteCarlo(env, state_n=state_n, action_n=action_n, epsilon_fun=calculate_epsilon, episode_n=2000, trajectory_len=1000, gamma=0.99))

def calculate_epsilon(episode, episode_n, last_reward, total_rewards):
    epsilon = min(0.1, 1 - 2*np.mean(total_rewards[:-10])/500)
    return epsilon

dict['min(0.1, 1 - 2*np.mean(total_rewards[:-10])/500)'] = (MonteCarlo(env, state_n=state_n, action_n=action_n, epsilon_fun=calculate_epsilon, episode_n=2000, trajectory_len=1000, gamma=0.99))

def calculate_epsilon(episode, episode_n, last_reward, total_rewards):
    epsilon = min(0.1, 1 - 2*np.mean(total_rewards[:-100])/500)
    return epsilon

dict['min(0.1, 1 - 2*np.mean(total_rewards[:-100])/500)'] = (MonteCarlo(env, state_n=state_n, action_n=action_n, epsilon_fun=calculate_epsilon, episode_n=2000, trajectory_len=1000, gamma=0.99))

window = 50

plot_results(dict)