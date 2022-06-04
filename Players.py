import random
from utils import *
from collections import defaultdict

class QLearningPlayer():
    def __init__(self, alpha=0.05, gamma=0.99, eps=0.2, decreasing_exploration=False, eps_min=0.1, eps_max=0.8, n_star=5000):
        self.alpha = alpha
        self.gamma = gamma
        self.decreasing_exploration = decreasing_exploration
        if decreasing_exploration:
            self.eps_min = eps_min
            self.eps_max = eps_max
            self.n_star = n_star
        else:
            self.eps = eps

        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))
        self.player = None

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

    def update_Q(self, state, action, next_state, reward):
        """ Update Q value """
        action = position_to_index(action)
        max_next_state = max(self.Q[next_state].values())
        self.Q[state][action] = (1 - self.alpha) * self.Q[state][action] + self.alpha * (reward + self.gamma * max_next_state)

    def act(self, grid):
        """ Play """
        if random.random() < self.eps:
            return self.randomMove(grid)
        else:
            avail = self.empty(grid)
            for index in position_to_index(avail):
                _ = self.Q[grid][index] # used to initialize all missing Q values
            best_action = max(self.Q[grid], key=self.Q[grid].get)
            return index_to_position(best_action)
        
