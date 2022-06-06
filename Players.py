import random
from collections import defaultdict, namedtuple, deque
import torch.nn as nn
import torch.nn.functional as F

from utils import *


class QLearningPlayer():
    """ Base class for Q-Learning. """
    def __init__(self, alpha=0.05, gamma=0.99, eps=0.2, decreasing_exploration=False, eps_min=0.1, eps_max=0.8, n_star=5000, player='X'):
        """
        :param alpha: learning rate
        :param gamma: discount factor
        :param eps: exploration level
        :param decreasing_exploration: enable decreasing exploration
        :param eps_min: minimum exploration level (for decreasing exploration)
        :param eps_max: maximum exploration level (for decreasing exploration)
        :param n_star: number of exploratory games (for decreasing exploration)
        :param player: 'X' or 'O' (can be set later using self.set_player())
        """
        self.alpha = alpha
        self.gamma = gamma
        self.decreasing_exploration = decreasing_exploration
        self.n = 0 # number of games done (used for decreasing exploration formula)
        self.eps = eps
        self.best_play = False # if True, set exploration level to 0 (used for metrics computations)
        self.player = player
        if decreasing_exploration:
            self.eps_min = eps_min
            self.eps_max = eps_max
            self.n_star = n_star

        # defaultdict of defaultdict containing 0.0 values (initial Q-values)
        # can be called like this: Q[state][action] with state being grid hashed (see grid_to_string()) and action being a int between 0-8 included
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))

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
        """
        return all empty positions
        :param grid: np.ndarray [3x3]
        :return: list of empty positions (tuples)
        """
        avail = []
        for i in range(9):
            pos = (int(i / 3), i % 3)
            if grid[pos] == 0:
                avail.append(pos)
        return avail

    def randomMove(self, grid):
        """
        Chose a random move from the available options.
        :param grid: np.ndarray [3x3]
        :return: random available position (tuple)
        """
        avail = self.empty(grid)
        return avail[random.randint(0, len(avail)-1)]

    def update_Q(self, new_grid, reward, grid, action):
        """
        Update Q value.
        :param new_grid: np.ndarray [3x3]
        :param reward: int (usually -1 or 0 or 1)
        :param grid: np.ndarray [3x3]
        :param action: position tuple (0-2 included, 0-2 included)
        :return: None
        """
        # input formatting
        action = position_to_index(action)
        grid = grid_to_string(grid)
        new_grid = grid_to_string(new_grid)

        _ = self.Q[grid][action] # used to initialize missing Q value
        if len(self.Q[new_grid].values()) != 0:
            max_next_state = max(self.Q[new_grid].values()) # get V(s_{t+1}, a)
        else:
            max_next_state = 0.0
        self.Q[grid][action] = (1 - self.alpha) * self.Q[grid][action] + self.alpha * (reward + self.gamma * max_next_state) # Q-Learning update formula

    def act(self, grid):
        """
        Return the best learned play on grid
        :param grid: np.ndarray [3x3]
        :return: position tuple (0-2 included, 0-2 included)
        """
        if self.decreasing_exploration:
            self.eps = max(self.eps_min, self.eps_max * (1 - self.n / self.n_star)) # decreasing exploration formula
        if random.random() < self.eps and self.best_play == False:
            move = self.randomMove(grid) # random move from available positions
        else:
            avail = self.empty(grid)
            for index in position_to_index(avail):
                _ = self.Q[grid_to_string(grid)][index] # used to initialize missing Q values
            # best_action = max(list(self.Q[grid_to_string(grid)]), key=self.Q[grid_to_string(grid)].get) # old implementation (not used anymore)
            actions = list(self.Q[grid_to_string(grid)].items()) # list of possible actions
            random.shuffle(actions) # shuffles the actions (to have even probabilities if multiple same maximal Q-values)
            best_action = max(actions, key=lambda x: x[1])[0] # get the best_action (highest Q-value)
            move = index_to_position(best_action) # transformation to position tuple
        return move


class BufferMemory(object):
    """ Buffer memory class, used for Deep Q-Learning. """
    def __init__(self, buffer_size):
        """
        First in - first out memory containing 'buffer_size' elements
        :param buffer_size: number of elements in BufferMemory (int)
        """
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # DEVICE used by PyTorch
        self.buffer = deque([], maxlen=buffer_size)  # first in - first out memory
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward')) # initialize Transition namedtuple generic object (namedtruple containing informations to update the networks)

    def __len__(self):
        """ Return the actual length of BufferMemory (used to check if enough data are present) """
        return len(self.buffer)

    def store(self, state, action, new_state, reward):
        """
        Stores a transition tuple containing (state, action, new_state, reward) data
        :param state: torch.Tensor [2x3x3 or Nx2x3x3]
        :param action: int (0-8) or torch.Tensor [1x1]
        :param new_state: None or torch.Tensor [2x3x3 or Nx2x3x3]
        :param reward: int or torch.Tensor [1x1]
        :return: None
        """
        # input formatting
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
        """
        Samples a random minibatch of transitions
        :param batch_size: int > 0
        :return: random minibatch of BufferMemory
        """
        return random.sample(self.buffer, batch_size)


class DQN(nn.Module):
    """ PyTorch network architecture for Deep Q-Learning. """
    def __init__(self):
        super(DQN, self).__init__()
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # DEVICE used by PyTorch

        self.lin1 = nn.Linear(18, 128).to(self.DEVICE)
        self.lin2 = nn.Linear(128, 128).to(self.DEVICE)
        self.lin3 = nn.Linear(128, 9).to(self.DEVICE)

    def forward(self, x_t):
        """
        Forward pass of the Deep Q-Learning Network.
        :param x_t: state, torch.Tensor [2x3x3 or Nx2x3x3]
        :return: action Q-values, torch.Tensor [Nx9]
        """
        x_t = x_t.to(self.DEVICE) # put the torch.Tensor to DEVICE
        if x_t.dim() == 3:
            x_t = x_t.view(1, x_t.shape[0], x_t.shape[1], x_t.shape[2])
        N = x_t.shape[0]

        x_t = F.relu(self.lin1(x_t.view(N, -1))) # hidden layer 1
        x_t = F.relu(self.lin2(x_t)) # hidden layer 2
        x_t = self.lin3(x_t) # output
        return x_t


class DeepQLearningPlayer(QLearningPlayer):
    """ Class for Deep Q-Learning. """
    def __init__(self, eps=0.2, decreasing_exploration=False, eps_min=0.1, eps_max=0.8, n_star=5000, buffer_size=10000, batch_size=64):
        """
        :param eps: exploration level
        :param decreasing_exploration: enable decreasing exploration
        :param eps_min: minimum exploration level (for decreasing exploration)
        :param eps_max: maximum exploration level (for decreasing exploration)
        :param n_star: number of exploratory games (for decreasing exploration)
        :param buffer_size: number of elements in BufferMemory (int)
        :param batch_size: minibatch size (int > 0)
        """
        super(DeepQLearningPlayer, self).__init__(eps=eps, decreasing_exploration=decreasing_exploration, eps_min=eps_min, eps_max=eps_max, n_star=n_star)
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # DEVICE used by PyTorch
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward')) # initialize Transition namedtuple generic object (namedtruple containing informations to update the networks)
        self.buffer = BufferMemory(buffer_size=buffer_size) # instantiate BufferMemory
        self.batch_size = batch_size
        self.learning_rate = 5e-4 # learning rate by default
        self.target_update = 500 # target_net updated each 'target_update' games

        self.policy_net = DQN().to(self.DEVICE) # Deep Q-Learning Network instantiation
        self.target_net = DQN().to(self.DEVICE) # target_net Network used as target for learning
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.save_loss = True # save the losses into self.losses
        self.losses = [] # store the losses
        self.average_loss = None # average the losses when get_loss_average called
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate) # PyTorch optimizer

        self.allow_illegal_random_move = False # allow illegal random move (always set to False)

    def get_loss_average(self):
        """
        Returns the average loss (from all the iterations since the last call of get_loss_average) and resets self.losses
        :return: average loss (float)
        """
        if len(self.losses) != 0:
            self.average_loss = sum(self.losses)/len(self.losses)
        else:
            self.average_loss = None
        self.losses = []
        return self.average_loss

    def optimize_model(self):
        """
        Optimization algorithm
        :return: None
        """
        # do not optimize model if not enough data
        if len(self.buffer) < self.batch_size:
            return None

        transitions = self.buffer.sample_random_minibatch(self.batch_size) # get minibatch from the MemoryBuffer
        batch = self.Transition(*zip(*transitions))  # converts list of Transitions to Transition of lists

        state_batch = torch.cat(batch.state) # concatenate states together into a torch.Tensor [batch_size x 2 x 3 x 3]
        action_batch = torch.cat(batch.action) # concatenate actions together into a torch.Tensor [batch_size x 1]
        reward_batch = torch.cat(batch.reward) # concatenate rewards together into a torch.Tensor [batch_size x 1]

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)  # predicts Q(s_t, a) using our model into a torch.Tensor [batch_size x 1]

        next_state_values = torch.zeros(self.batch_size, device=self.DEVICE)  # initialize V(s_{t+1}) to 0.0 into a torch.Tensor [batch_size x 1]
        next_state_empty = (sum(x is not None for x in batch.next_state) == 0)  # bool to check if next_state is empty
        if next_state_empty == False:
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.DEVICE, dtype=torch.bool)  # mask of non-final states
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])  # next_states of non-final states
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        next_state_values = next_state_values.unsqueeze(1) # adds a dimension

        expected_state_action_values = (reward_batch + self.gamma * next_state_values) # expected Q(s_t, a)

        # criterion = nn.HuberLoss(delta=1.0).to(self.DEVICE) # Huber Loss
        criterion = nn.SmoothL1Loss().to(self.DEVICE) # equals to Huber Loss (works for lower version of PyTorch)
        loss = criterion(state_action_values, expected_state_action_values) # calculate loss
        if self.save_loss: self.losses.append(loss.to('cpu').item()) # if save_loss, store the loss into self.losses

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1) # clamping of grads
        self.optimizer.step()

    def act(self, grid):
        """
        Return the best learned play on grid
        :param grid: np.ndarray [3x3] or torch.Tensor [2x3x3]
        :return: position index (0-8 included)
        """
        if type(grid) is np.ndarray:
            grid = grid_to_tensor(grid, self.player)
        if self.decreasing_exploration:
            self.eps = max(self.eps_min, self.eps_max * (1 - self.n / self.n_star)) # decreasing exploration formula
        if random.random() < self.eps and self.best_play == False:
            if self.allow_illegal_random_move:
                move = torch.tensor([[random.randrange(9)]], device=self.DEVICE) # if illegal moves allowed
            else:
                move = torch.tensor([[position_to_index(self.randomMove(tensor_to_grid(grid, self.player)))]], device=self.DEVICE) # random move from available positions
        else:
            with torch.no_grad():
                move = self.policy_net(grid).max(dim=1)[1].view(-1, 1) # get the best possible action (highest Q-value outputed by the network even if illegal move)
        move = move.view(-1).tolist() # from torch.Tensor to list
        if len(move) == 1: move = move[0] # if only one element
        return move

