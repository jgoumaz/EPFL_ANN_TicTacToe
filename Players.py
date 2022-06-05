import random
from utils import *
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.nn.functional as F

class QLearningPlayer():
    def __init__(self, alpha=0.05, gamma=0.99, eps=0.2, decreasing_exploration=False, eps_min=0.1, eps_max=0.8, n_star=5000, player='X'):
        self.alpha = alpha
        self.gamma = gamma
        self.decreasing_exploration = decreasing_exploration
        self.n = 0
        self.eps = eps
        self.best_play = False
        if decreasing_exploration:
            self.eps_min = eps_min
            self.eps_max = eps_max
            self.n_star = n_star

        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.player = player

    def set_player(self, player='X', j=-1):
        """
        Set the player (must be called after instanciating the class).
        :param player: 'X' or 'O', defaults to 'X'
        :type player: str, optional
        :param j: even number for X and odd number for 'O', defaults to -1 (j not used)
        :type j: int, optional
        """
        self.player = player
        if j != -1:
            self.player = 'X' if j % 2 == 0 else 'O'

    def empty(self, grid):
        """ return all empty positions """
        avail = []
        for i in range(9):
            pos = (int(i / 3), i % 3)
            if grid[pos] == 0:
                avail.append(pos)
        return avail

    def randomMove(self, grid):
        """ Chose a random move from the available options. """
        avail = self.empty(grid)

        return avail[random.randint(0, len(avail)-1)]

    def update_Q(self, new_grid, reward, grid, action):
        """ Update Q value """
        action = position_to_index(action)
        grid = grid_to_string(grid)
        new_grid = grid_to_string(new_grid)
        _ = self.Q[grid][action] # used to initialize missing Q value
        if len(self.Q[new_grid].values()) != 0:
            max_next_state = max(self.Q[new_grid].values())
        else:
            max_next_state = 0.0
        self.Q[grid][action] = (1 - self.alpha) * self.Q[grid][action] + self.alpha * (reward + self.gamma * max_next_state)

    def act(self, grid):
        """ Play """
        if self.decreasing_exploration:
            self.eps = max(self.eps_min, self.eps_max * (1 - self.n / self.n_star))
        if random.random() < self.eps and self.best_play == False:
            move = self.randomMove(grid)
        else:
            avail = self.empty(grid)
            for index in position_to_index(avail):
                _ = self.Q[grid_to_string(grid)][index] # used to initialize all missing Q values
            # best_action = max(list(self.Q[grid_to_string(grid)]), key=self.Q[grid_to_string(grid)].get)
            actions = list(self.Q[grid_to_string(grid)].items())
            random.shuffle(actions)
            best_action = max(actions, key=lambda x: x[1])[0]
            move = index_to_position(best_action)
        return move


class BufferMemory(object):
    def __init__(self, buffer_size):
        self.buffer = deque([], maxlen=buffer_size)

    def __len__(self):
        return len(self.buffer)

    def store(self, transition):
        """ Stores a transition containing (new_state, reward, state, action) """
        self.buffer.append(transition)

    def sample_random_minibatch(self, batch_size):
        """ Samples a random minibatch of transitions """
        return random.sample(self.buffer, batch_size)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.lin1 = nn.Linear(18, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, 9)

    def forward(self, x_t):
        x_t = x_t.to(self.DEVICE)
        if x_t.dim() == 3:
            x_t = x_t.view(1, x_t.shape[0], x_t.shape[1], x_t.shape[2])
        N = x_t.shape[0]

        x_t = F.relu(self.lin1(x_t.view(N, -1)))
        x_t = F.relu(self.lin2(x_t))
        x_t = self.lin3(x_t)
        return x_t