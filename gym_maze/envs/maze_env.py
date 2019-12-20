import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np


class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.grid = np.zeros((9, 9))
        self.m = 9
        self.n = 9
        self.state_space = [i for i in range(self.m*self.n)]
        self.state_space_size = [i for i in range(self.m*self.n)]
        self.action_space = [0, 1, 2, 3]
        self.agent_position = 0
        self.reward = 0
        self.done = False
        self.placeholder = "hi"
        self.step_count = 0
        self.enemy_list = (2, 3, 6, 15, 17, 19, 24, 26, 28, 29, 31, 33, 35, 37, 40, 42, 46, 49, 55, 57, 58, 61, 77, 78)

# 0 ia right, 1 is left, 2 is up, 3 is down

    def step(self, action):
        if action == 0:
            if (self.agent_position+1) % 9 == 0:
                self.agent_position = self.agent_position
            else:
                self.agent_position += 1

        elif action == 1:
            if self.agent_position % 9 == 0:
                self.agent_position = self.agent_position
            else:
                self.agent_position -= 1

        elif action == 2:
            if self.agent_position < 9:
                self.agent_position = self.agent_position
            else:
                self.agent_position -= 9

        elif action == 3:
            if self.agent_position >= 72:
                self.agent_position = self.agent_position
            else:
                self.agent_position += 9

        self.step_count += 1
        if self.step_count == 300:
            self.done = True
        self.get_reward()

        return [self.agent_position, self.reward, self.done, self.placeholder]

    def get_reward(self):
        if self.agent_position == 80:
            self.reward = 300
            self.done = True

        elif self.agent_position in self.enemy_list:
            self.reward = -25
        elif self.agent_position not in self.enemy_list:
            self.reward = -1

    def reset(self):
        self.agent_position = 0
        self.reward = 0
        self.grid = np.zeros((9, 9))
        self.done = False
        self.step_count = 0
        return self.agent_position

    def render(self, mode='human', close=False):
        self.grid = np.zeros((9, 9))
        self.grid[self.agent_position // 9, self.agent_position % 9] = 1
        print(self.grid)
        print(self.agent_position)

    def action_space_sample(self):
        return np.random.choice(self.action_space)





