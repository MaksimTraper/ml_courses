import torch
import torch.nn as nn
import gym
import time
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCarContinuous-v0')
state_dim = 2
action_n = env.action_space.shape[0]

class CEM(nn.Module):
    def __init__(self, state_dim, action_n, epsilon=0, epsilon_decrease=1e-3, epsilon_min=1e-2, hidden_size=128, lr=1e-3):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n
        self.network = nn.Sequential(nn.Linear(self.state_dim, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, self.action_n)
                                    )
        self.optimazer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.epsilon = epsilon
        self.epsilon_descrease = epsilon_decrease
        self.epsilon_min = epsilon_min

    def forward(self, _input):
        return self.network(_input)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        action = self.forward(state).detach().numpy() + np.random.normal(0, self.epsilon, size=self.action_n)
        # Ограничим действие значениями от -1 до 1
        action = np.clip(action, env.action_space.low, env.action_space.high)
        return action

    def fit(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                elite_states.append(state)
                elite_actions.append(action)
        elite_states = torch.FloatTensor(elite_states)
        elite_actions = torch.LongTensor(elite_actions)
        pred_actions = self.forward(elite_states)

        loss = self.loss(pred_actions, elite_actions)
        loss.backward()
        self.optimazer.step()
        self.optimazer.zero_grad()

        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_descrease)

def get_trajectory(env, agent, max_len=1000, visualize=False):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    state = env.reset()

    for _ in range(max_len):
        trajectory['states'].append(state)

        action = agent.get_action(state)
        trajectory['actions'].append(action)

        state, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)

        if visualize:
            time.sleep(0.5)
            env.render()

        if done:
            break

    return trajectory

#plt.figure(figsize=(10, 12))
plt.grid(True)
plt.title('CEM при конечномерном пространстве действий')
plt.ylabel('Средняя награда', fontsize = 'medium')
plt.xlabel('Итерация', fontsize = 'medium')

#q_param = [0.2, 0.4, 0.6, 0.8, 0.9]
q_param = [0.8]
iteration_ns = [20]
#trajectory_ns = [10, 20, 50]
trajectory_ns = [50]
#trajectory_lens = [300, 500, 1000]
trajectory_lens = [300]
epsilons = [0.3]
#epsilons = [0.1, 0.3, 0.5, 0.7, 0.9]
hidden_sizes = [64, 128, 256]
lrs = [1e-3]

for q in q_param:
  for iteration_n in iteration_ns:
    for trajectory_n in trajectory_ns:
      for trajectory_len in trajectory_lens:
        for epsilon in epsilons:
            for hidden_size in hidden_sizes:
                for lr in lrs:

                    agent = CEM(state_dim, action_n, epsilon_decrease=1/iteration_n, epsilon=epsilon, hidden_size=hidden_size, lr=lr)
                    mean_total_rewards = []
                    for iteration in range(iteration_n):
                        #policy evaluation
                        trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
                        total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
                        print('iteration:', iteration, 'mean total reward:', np.mean(total_rewards))

                        mean_total_rewards.append(np.mean(total_rewards))

                        #policy improvement
                        quantile = np.quantile(total_rewards, q)
                        elite_trajectories = []
                        for trajectory in trajectories:
                            total_reward = np.sum(trajectory['rewards'])
                            if total_reward > quantile:
                                elite_trajectories.append(trajectory)

                        if len(elite_trajectories) > 0:
                            agent.fit(elite_trajectories)

                    trajectory = get_trajectory(env, agent, max_len=trajectory_len, visualize=False)
                    print('total reward:', sum(trajectory['rewards']))
                    plt.plot(mean_total_rewards, label=f'q = {q}, iteration_n = {iteration_n}, trajectory_n = {trajectory_n}, trajectory_len = {trajectory_len}, epsilon = {epsilon}, hidden_size = {hidden_size}, lr = {lr}')

plt.legend(loc='lower left', fontsize='small')
plt.show()