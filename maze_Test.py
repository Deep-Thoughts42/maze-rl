import gym
import gym_maze
import numpy as np

env = gym.make('Maze-v0')


LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2500
SHOW_EVERY = 100
epsilon = 0.5
START_EPS_DECAY = 1
STATS_EVERY = 50
END_EPS_DECAY = EPISODES//2
epsilon_decay_value = epsilon//(END_EPS_DECAY - START_EPS_DECAY)

done = False

q_table = np.zeros((81, 4))

# episode_rewards = []
# aggr_ep_rewards = {'ep': [], 'avg': [], 'max': [], 'min': []}

for episode in range(EPISODES):
    state = env.reset()
    done = False

    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False

    while not done:

        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[state])
        else:
            # Get random action
            action = np.random.randint(0, 4)

        new_state, reward, done, _ = env.step(action)

        new_state = new_state

        if (episode % SHOW_EVERY) == 0:
            env.render()

        # If simulation did not end yet after last step - update Q table
        if not done:

            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_state, :])

            # Current Q value (for current state and performed action)
            current_q = q_table[state, action]

            # And here's our equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            q_table[state, action] = new_q


        state = new_state

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPS_DECAY >= episode >= START_EPS_DECAY:
        epsilon -= epsilon_decay_value


env.close()

