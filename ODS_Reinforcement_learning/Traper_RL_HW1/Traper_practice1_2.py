import gym
import gym_maze
import numpy as np
import random
import time
import matplotlib.pyplot as plt

class RandomAgent():
    def __init__(self, action_n):
        self.action_n = action_n

    # Метод для произвольного движения агента по лабиринту    
    def get_action(self, state):
        action = np.random.randint(self.action_n)
        return action


class CrossEntropyAgent():
    def __init__(self, state_n, action_n):
        self.state_n = state_n
        self.action_n = action_n

        # задаём изначальную политику в виде двумерной матрицы с 
        # равновероятностными действиями для каждого состояния
        self.model = np.ones((self.state_n, self.action_n)) / self.action_n

    def get_action(self, state):
        action = np.random.choice(np.arange(self.action_n), p=self.model[state])
        return int(action)

    # Функция обучения новой "эффективной" политики
    def fit(self, elite_trajectories, smoothing, coef_lambda = 0.1):
        new_model = np.zeros((self.state_n, self.action_n))
        # считаем матрицу состояния/действия для всех элитных траекторий
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1

                if smoothing == 'Laplace':
                    new_model[state][action] += coef_lambda

        for state in range(self.state_n):
            # если модель в рамках какой-то траектории была в очередном состоянии
            # то вычисляем вероятность для каждого действия в этом состоянии
            if np.sum(new_model[state]) > 0:
                if smoothing == 'Laplace':
                    if(coef_lambda > 0):
                        new_model[state] /= (np.sum(new_model[state]) + coef_lambda * action_n)

                        # вероятность в строке без этого не всегда = 1
                        new_model[state] /= np.sum(new_model[state])
                    else:
                        print('Выбран неправильный коэффициент лямбда для сглаживания Лапласа')
                        return
                else:
                    new_model[state] /= np.sum(new_model[state])

                if smoothing == 'Policy':
                    if(0 < coef_lambda <= 1):
                        new_model[state] = coef_lambda * new_model[state] + (1 - coef_lambda) * self.model[state].copy()
                    else:
                        print('Выбран неправильный коэффициент лямбда для сглаживания политики')
                        return

            # если сумма по строке == 0
            else:
                new_model[state] = self.model[state].copy()
        self.model = new_model
        return None

# Метод получения № ячейки (состояния), в которой находится агент
def get_state(obs):
    return int(np.sqrt(state_n) * obs[0] + obs[1])

# Сэмплируем случайную траекторию
def get_trajectory(env, agent, trajectory_len=1000, visualize=False):
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

        if visualize:
            _, _, passenger_location, destination = env.decode(state)
            locations = {0: 'Red', 1: 'Green', 2: 'Yellow', 3: 'Blue', 4: 'In taxi'}
            print(f'Passenger_location = {locations[passenger_location]}; destination = {locations[destination]}')

            time.sleep(0.2)
            env.render()

        if done:
            break
    
    return trajectory

env = gym.make('Taxi-v3')
# Количество состояний
state_n = 500
# Количество действия
action_n = 6

def quantile_scheduler(num_iter, q_param = 0.4, operation='Need_plus'):
    if (operation == 'Need_plus'):
        add = 0.2 * ((num_iter + 1) // 50)
        return q_param + add if q_param + add < 1 else 0.95
    else:
        minus = 0.2 * ((num_iter + 1) // 50)
        return q_param - minus if q_param - minus > 0 else 0.1

def fit_agent(q_param = 0.9, iteration_n = 20, trajectory_n = 50, need_schedule=False, smoothing=False, coef_lambda = 0.1):
    agent = CrossEntropyAgent(state_n, action_n)
    mean_total_rewards = []

    for iteration in range(iteration_n):

        #policy evaluation
        trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
        total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
        print('iteration:', iteration, 'mean total reward:', np.mean(total_rewards))

        mean_total_rewards.append(np.mean(total_rewards))

        if (need_schedule != False):
            new_q_param = quantile_scheduler(iteration, q_param, need_schedule)
        else:
            new_q_param = q_param

        #policy improvement
        quantile = np.quantile(total_rewards, new_q_param)
        elite_trajectories = []
        for trajectory, total_reward in zip(trajectories, total_rewards):
            if total_reward > quantile:
                elite_trajectories.append(trajectory)

        agent.fit(elite_trajectories, smoothing, coef_lambda)

    trajectory = get_trajectory(env, agent, trajectory_len=100, visualize=True)
    print('total reward:', sum(trajectory['rewards']))
    print('model:')
    print(agent.model)

    plt.plot(mean_total_rewards, label=f'q = {q_param}, iteration_n = {iteration_n}, trajectory_n = {trajectory_n}, need_scheduler = {need_scheduler}, smoothing={smoothing}, lambda = {coef_lambda}')

iterations_n = [100]
#iterations_n = [20, 50, 100, 150]

trajectorys_n = [300]
#trajectorys_n = [50]

#if_need_scheduler = ['Need_plus', 'Need_minus', False]
if_need_scheduler = [False]

qs = [0.4]

#qs = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]

smoothings = [False, 'Laplace', 'Policy']
coef_lambda = 0.95

for smoothing in smoothings:
    for need_scheduler in if_need_scheduler:
        for q in qs:
            for iteration_n in iterations_n:
                for trajectory_n in trajectorys_n:
                    if need_scheduler == "Need_minus":
                        q = 0.9
                    else:
                        q = 0.4

                    print(f'q = {q}, iteration_n = {iteration_n}, trajectory_n = {trajectory_n}, need_scheduler = {need_scheduler}, smoothing={smoothing}, lambda = {coef_lambda}')

                    fit_agent(q, iteration_n, trajectory_n, need_scheduler, smoothing, coef_lambda)

plt.ylabel('Средняя награда')
plt.xlabel('Итерация')
plt.legend(loc='best')
plt.show()