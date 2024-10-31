import numpy as np
import time
from Frozen_Lake import FrozenLakeEnv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

env = FrozenLakeEnv()

def get_q_values(values, gamma):
    q_values = {}
    for state in env.get_all_states():
        q_values[state] = {}
        for action in env.get_possible_actions(state):
            q_values[state][action] = 0
            for next_state in env.get_next_states(state, action):
                q_values[state][action] += env.get_transition_prob(state, action, next_state) * (env.get_reward(state, action, next_state) + gamma * values[next_state])
    return q_values

def init_policy():
    policy = {}
    for state in env.get_all_states():
        policy[state] = {}
        for action in env.get_possible_actions(state):
            policy[state][action] = 1 / len(env.get_possible_actions(state))
    return policy

def init_values():
    values = {}
    for state in env.get_all_states():
        values[state] = 0
    return values

def calculate_new_values(policy, gamma, L, values=None):
    for l in range(L):
        new_values = init_values()
        for state in env.get_all_states():
            for action in env.get_possible_actions(state):
                q_values = get_q_values(values, gamma)
                new_values[state] += policy[state][action] * q_values[state][action]
        values = new_values
    return values


def policy_evaluation(values, gamma):
    return get_q_values(values, gamma)

def policy_improvement(q_values):
    policy = init_policy()
    for state in env.get_all_states():
        if len(env.get_possible_actions(state)) > 0:
            max_i = np.argmax(list(q_values[state].values()))
            max_action = env.get_possible_actions(state)[max_i]
            for action in env.get_possible_actions(state):
                policy[state][action] = 1 if action == max_action else 0
    return policy

#Ks = [1, 3, 5, 10, 20, 30]
Ks = [20]
#L = [20]
Ls = [3, 5, 10, 20, 30, 50]
gamma = 0.995
df = pd.DataFrame(columns=['K', 'L', 'result', 'new_val'])

for i in range(5):
    for new_val in ['True', 'False']:
        for L in Ls:
            for K in Ks:
                policy = init_policy()
                values = init_values()  # Инициализируем values только один раз перед K-итерациями

                for k in range(K):
                    if new_val == 'True':
                        # Используем значения values, накопленные на предыдущем шаге
                        values = calculate_new_values(policy, gamma, L, values)
                        q_values = policy_evaluation(values, gamma)
                    else:
                        # Начинаем с нулевых значений values на каждом шаге
                        values = init_values()
                        values = calculate_new_values(policy, gamma, L, values)
                        q_values = policy_evaluation(values, gamma)

                    # Обновляем policy и сохраняем обновленные values для следующей итерации
                    policy = policy_improvement(q_values)

                total_rewards = []
                for _ in range(1000):
                    state = env.reset()
                    total_reward = 0
                    for i in range(1000):
                        action_i = np.random.choice(np.arange(4), p=list(policy[state].values()))
                        action = env.get_possible_actions(state)[action_i]
                        state, reward, done, _ = env.step(action)
                        total_reward += reward
                        if done:
                            break
                    total_rewards.append(total_reward)
                print(f'gamma = {gamma}, result = {np.mean(total_rewards)}')
                df.loc[len(df.index)] = {'K': K, 'L': L, 'result': np.mean(total_rewards), 'new_val': new_val}

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='L', y='result', hue='new_val', s=100)
plt.xticks(rotation=45, fontsize=12) 
plt.yticks(fontsize=12)
plt.xlabel('L', fontsize=14) 
plt.ylabel('Result', fontsize=14)
plt.title('Как влияет работа с values на результаты', fontsize=16)
plt.legend(title='New step - new val', title_fontsize=12, fontsize=10) 

plt.show()