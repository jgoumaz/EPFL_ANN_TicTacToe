import random
from utils import *
from collections import defaultdict, namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.buffer = deque([], maxlen=buffer_size)
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    def __len__(self):
        return len(self.buffer)

    def store(self, state, action, new_state, reward):
        """ Stores a transition containing (state, action, new_state, reward) """
        if type(action) is not torch.Tensor:
            action = torch.tensor([[action]]).to(self.DEVICE)
        if type(reward) is not torch.Tensor:
            reward = torch.tensor([[reward]]).to(self.DEVICE)
        if state.ndim == 3:
            state = state.unsqueeze(0)
        if new_state is not None:
            if new_state.ndim == 3:
                new_state = new_state.unsqueeze(0)
        self.buffer.append(self.Transition(state, action, new_state, reward))

    def sample_random_minibatch(self, batch_size):
        """ Samples a random minibatch of transitions """
        return random.sample(self.buffer, batch_size)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.lin1 = nn.Linear(18, 128).to(self.DEVICE)
        self.lin2 = nn.Linear(128, 128).to(self.DEVICE)
        self.lin3 = nn.Linear(128, 9).to(self.DEVICE)

    def forward(self, x_t):
        x_t = x_t.to(self.DEVICE)
        if x_t.dim() == 3:
            x_t = x_t.view(1, x_t.shape[0], x_t.shape[1], x_t.shape[2])
        N = x_t.shape[0]

        x_t = F.relu(self.lin1(x_t.view(N, -1)))
        x_t = F.relu(self.lin2(x_t))
        x_t = self.lin3(x_t)
        return x_t


class DeepQLearningPlayer(QLearningPlayer):
    def __init__(self, eps=0.2, decreasing_exploration=False, eps_min=0.1, eps_max=0.8, n_star=5000):
        super(DeepQLearningPlayer, self).__init__(eps=eps, decreasing_exploration=decreasing_exploration, eps_min=eps_min, eps_max=eps_max, n_star=n_star)
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        self.buffer = BufferMemory(10000)
        self.batch_size = 64
        self.learning_rate = 5e-4
        self.target_update = 500

        self.policy_net = DQN().to(self.DEVICE)
        self.target_net = DQN().to(self.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.save_loss = True
        self.losses = []
        self.average_loss = None
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

    def get_loss_average(self):
        if len(self.losses) != 0:
            self.average_loss = sum(self.losses)/len(self.losses)
        else:
            self.average_loss = None
        self.losses = []
        return self.average_loss

    def optimize_model(self):
        # do not optimize model if not enough data
        if len(self.buffer) < self.batch_size:
            return None

        transitions = self.buffer.sample_random_minibatch(self.batch_size)
        batch = self.Transition(*zip(*transitions))  # converts list of Transitions to Transition of lists

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.DEVICE, dtype=torch.bool)  # mask of non-final states
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])  # next states of non-final states
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)  # Q(s_t, a)

        next_state_values = torch.zeros(self.batch_size, device=self.DEVICE)  # V(s_{t+1})
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        next_state_values = next_state_values.unsqueeze(1)

        # print(state_batch.shape, action_batch.shape, reward_batch.shape, next_state_values.shape)
        expected_state_action_values = (reward_batch + self.gamma * next_state_values)  # expected Q(s_t, a)

        # print(state_action_values.shape, expected_state_action_values.shape)

        criterion = nn.HuberLoss(delta=1.0).to(self.DEVICE)
        loss = criterion(state_action_values, expected_state_action_values)
        if self.save_loss: self.losses.append(loss.to('cpu').item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    # def randomMove(self, grid):
    #     """ Chose a random move. """
    #     return torch.tensor([[random.randrange(9)]], device=self.DEVICE) # FIXME: dtype=torch.long ?

    def act(self, grid):
        """ Play """
        if type(grid) is np.ndarray:
            grid = grid_to_tensor(grid, self.player)
        if self.decreasing_exploration:
            self.eps = max(self.eps_min, self.eps_max * (1 - self.n / self.n_star))
        if random.random() < self.eps and self.best_play == False:
            # move = self.randomMove(grid)
            move = torch.tensor([[position_to_index(self.randomMove(tensor_to_grid(grid, self.player)))]], device=self.DEVICE)
        else:
            with torch.no_grad():
                move = self.policy_net(grid).max(dim=1)[1].view(-1, 1)
        move = move.view(-1).tolist()
        if len(move) == 1: move = move[0]
        return move

