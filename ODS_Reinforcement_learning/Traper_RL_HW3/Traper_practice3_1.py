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

def policy_evaluation(policy, gamma, L):
    values = init_values()
    for l in range(L):
        new_values = init_values()
        for state in env.get_all_states():
            for action in env.get_possible_actions(state):
                q_values = get_q_values(values, gamma)
                new_values[state] += policy[state][action] * q_values[state][action]
        values = new_values
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

K = 20
L = 20
#gamma = 0.995
gammas = [0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
df = pd.DataFrame(columns=['gamma', 'result'])

for gamma in gammas:
    for i in range(3):
        policy = init_policy()
        for k in range(K):
            q_values = policy_evaluation(policy, gamma, L)
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
        df.loc[len(df.index)] = {'gamma': gamma, 'result': np.mean(total_rewards)}

df['gamma'] = df['gamma'].astype(str)

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='gamma', y='result', s=100)  # s=100 увеличивает размер точек для лучшей видимости
plt.xticks(rotation=45)  # Поворачиваем метки оси x для удобства чтения
plt.xlabel('Gamma')  # Подпись оси x
plt.ylabel('Result')  # Подпись оси y
plt.title('Зависимость результата от значения gamma')  # Заголовок графика
plt.show()