import gym
import gym_maze
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Maze-v0')


LEARNING_RATE = 0.1
DISCOUNT = 1
epsilon = 1.0
STATS_EVERY = 100

q_table = {}
for state in env.state_space:
    for action in env.action_space:
        q_table[state, action] = 0

num_games = 10000
totalRewards = np.zeros(num_games)


def max_action(q, state, actions):
    values = np.array([q[state, a] for a in actions])
    action = np.argmax(values)
    return actions[action]


for i in range(num_games):

    done = False
    epRewards = 0
    observation = env.reset()

    while not done:
        rand = np.random.random()
        action = max_action(q_table, observation, env.action_space) if rand < (1-epsilon) \
            else env.action_space_sample()
        observation_, reward, done, info = env.step(action)
        epRewards += reward
        action_ = max_action(q_table, observation_, env.action_space)
        q_table[observation, action] = q_table[observation, action] + LEARNING_RATE*(reward + DISCOUNT*q_table[observation_, action_] - q_table[observation, action])
        observation = observation_

    if epsilon - 2 / num_games > 0:
        epsilon -= 2/num_games
    else:
        epsilon = 0
    totalRewards[i] = epRewards

    if i % STATS_EVERY == 0:
        print('starting game', i)


plt.plot(totalRewards)
plt.xlabel("Number of Attempts")
plt.ylabel('Attempt Reward')
plt.title("Attempt Reward vs. Number of Attempts")

plt.show()




