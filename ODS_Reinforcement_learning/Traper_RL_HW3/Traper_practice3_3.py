import numpy as np
import time
from Frozen_Lake import FrozenLakeEnv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

env = FrozenLakeEnv()

def Policy_iteration(Ks = 20, Ls = 20, gammas = [0.995], new_val = False):
    global env_calls
    env_calls = 0

    def get_q_values(values, gamma):
        global env_calls
        q_values = {}
        for state in env.get_all_states():
            env_calls += 1
            q_values[state] = {}
            for action in env.get_possible_actions(state):
                env_calls += 1
                q_values[state][action] = 0
                for next_state in env.get_next_states(state, action):
                    env_calls += 1
                    q_values[state][action] += env.get_transition_prob(state, action, next_state) * (env.get_reward(state, action, next_state) + gamma * values[next_state])
                    env_calls += 2
        return q_values

    def init_policy():
        global env_calls
        policy = {}
        for state in env.get_all_states():
            env_calls += 1
            policy[state] = {}
            for action in env.get_possible_actions(state):
                env_calls += 1
                policy[state][action] = 1 / len(env.get_possible_actions(state))
                env_calls += 1
        return policy

    def init_values():
        global env_calls
        values = {}
        for state in env.get_all_states():
            env_calls += 1
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
        global env_calls
        policy = init_policy()
        for state in env.get_all_states():
            env_calls += 1
            if len(env.get_possible_actions(state)) > 0:
                env_calls += 1
                max_i = np.argmax(list(q_values[state].values()))
                max_action = env.get_possible_actions(state)[max_i]
                env_calls += 1
                for action in env.get_possible_actions(state):
                    env_calls += 1
                    policy[state][action] = 1 if action == max_action else 0
        return policy

    df = pd.DataFrame(columns=['env_calls', 'gamma', 'result'])

    for gamma in gammas:
        for K in Ks:
            for L in Ls:
                for new_val in ['True', 'False']:
                    for i in range(5):
                        env_calls = 0
                        policy = init_policy()
                        values = init_values() 
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
                        df.loc[len(df.index)] = {'env_calls': env_calls, 'gamma': gamma, 'result': np.mean(total_rewards)}

    return df

def Value_iteration(gammas=[0.9995], Ks=[20]):
    global env_calls
    env_calls = 0

    def init_values():
        global env_calls
        values = {}
        for state in env.get_all_states():
            env_calls += 1
            values[state] = 0
        return values

    def value_iteration(gamma, K):
        global env_calls
        values = init_values()
        best_actions = {}
        for k in range(K):
            new_values = values.copy()
            for state in env.get_all_states():
                env_calls += 1
                q_values = {}
                for action in env.get_possible_actions(state):
                    env_calls += 1
                    q_value = 0
                    for next_state in env.get_next_states(state, action):
                        env_calls += 1
                        q_value += env.get_transition_prob(state, action, next_state) * (
                            env.get_reward(state, action, next_state) + gamma * values[next_state]
                        )
                        env_calls += 2
                    q_values[action] = q_value
                if q_values:
                    best_action = max(q_values, key=q_values.get)
                    best_action_value = q_values[best_action]
                    new_values[state] = best_action_value
                    best_actions[state] = best_action
            values = new_values
        return best_actions

    def extract_policy(best_actions):
        global env_calls
        policy = {}
        for state in env.get_all_states():
            env_calls += 1
            if state in best_actions:
                policy[state] = best_actions[state]
            else:
                policy[state] = None
        return policy

    df = pd.DataFrame(columns=['env_calls', 'K', 'gamma', 'result'])

    for K in Ks:
        for gamma in gammas:
            for i in range(5):
                env_calls = 0
                best_actions = value_iteration(gamma, K)
                policy = extract_policy(best_actions)

                total_rewards = []
                for _ in range(1000):
                    state = env.reset()
                    total_reward = 0
                    for i in range(1000):
                        action = policy[state]
                        if action is None:
                            break
                        state, reward, done, _ = env.step(action)
                        total_reward += reward
                        if done:
                            break
                    total_rewards.append(total_reward)
                print(f'K = {K}, gamma = {gamma}, result = {np.mean(total_rewards)}')
                df.loc[len(df.index)] = {'env_calls': env_calls, 'K': K, 'gamma': gamma, 'result': np.mean(total_rewards)}

    return df

#gammas = [0.1, 0.2, 0.4, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999]
gammas = [0.999]
K = [100]
#K = [5, 10, 20, 30, 50, 75, 100, 200]
global env_calls
env_calls = 0
df_VI = Value_iteration(gammas, K)

gammas = [0.999]
K = [100]
L = [20]

df_VI['gamma'] = df_VI['gamma'].astype(str)
df_VI['K'] = df_VI['K'].astype(str)
df_VI['env_calls'] = df_VI['env_calls'].astype(str)

df_PI_F = Policy_iteration(Ks=K, Ls=L, gammas=gammas, new_val=False)
df_PI_F['env_calls'] = df_PI_F['env_calls'].astype(str)

df_PI_T = Policy_iteration(Ks=K, Ls=L, gammas=gammas, new_val=True)
df_PI_T['env_calls'] = df_PI_T['env_calls'].astype(str)

y_min, y_max = round(df_PI_F['result'].min(), 2), 1
y_ticks = np.arange(y_min, y_max, 0.02)

plt.figure(figsize=(12, 6))
"""
if len(gammas) > 1:
    sns.scatterplot(data=df_VI, x='gamma', y='result', s=200)
else:
    sns.scatterplot(data=df_VI, x='K', y='result', s=200) 
"""

sns.scatterplot(data=df_VI, x='env_calls', y='result', s=200, label="Value Iteration")
sns.scatterplot(data=df_PI_F, x='env_calls', y='result', s=200, label="Policy Iteration. New_val = False")
sns.scatterplot(data=df_PI_T, x='env_calls', y='result', s=200, label="Policy Iteration. New_val = True")

plt.xticks(rotation=45, fontsize=12) 
plt.yticks(y_ticks, fontsize=12)
"""
if len(gammas) > 1:
    plt.xlabel('Gamma', fontsize=14)
else:
    plt.xlabel('K', fontsize=14)
"""
plt.xlabel('Env_calls', fontsize=14)
plt.ylabel('Result', fontsize=14)
plt.title('Количество обращений к среде разных алгоритмов')

# Отображение легенды
plt.legend(title='Алгоритм', title_fontsize=12, fontsize=10)
plt.grid()
plt.show()