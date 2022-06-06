import warnings
import torch
import numpy as np


def index_to_position(index):
    """
    Transform 1d int index to 2d tuple position
    :param index: int (0-8 included)
    :return: tuple (0-2 included, 0-2 included)
    """
    if type(index) is int:
        position = (int(index / 3), index % 3)
        return position
    elif type(index) is list:
        positions = []
        for el in index:
            position = (int(el / 3), el % 3)
            positions.append(position)
        return positions
    else:
        warnings.warn("index_to_position got an unexpected input.")
        return None


def position_to_index(position):
    """
    Transform 2d tuple position to 1d int index
    :param position: tuple (0-2 included, 0-2 included)
    :return: int (0-8 included)
    """
    if type(position[0]) is int:
        index = 3 * position[0] + position[1]
        return index
    elif type(position[0]) is tuple:
        indices = []
        for el in position:
            index = 3 * el[0] + el[1]
            indices.append(index)
        return indices
    else:
        warnings.warn("position_to_index got an unexpected input.")
        return None


def grid_to_string(grid):
    """
    Get the 'hash' of a grid (used as keys for dictionaries of Q-values)
    Example: '-O-XO-X--' for |- O -|
                             |X O -|
                             |X - -|
    :param grid: np.ndarray [3x3]
    :return: string of 9 characters (hash of grid)
    """
    grid_flatten = list(grid.astype(int).flatten())
    grid_flatten_converted = [str(el+1).replace('0','O').replace('1','-').replace('2','X') for el in grid_flatten]
    grid_hash = "".join(grid_flatten_converted)
    return grid_hash


def grid_to_tensor(grid, player='X'):
    """
    Transform a 3x3 grid to a 2x3x3 torch.Tensor
    :param grid: np.ndarray [3x3]
    :param player: 'X' or 'O'
    :return: torch.Tensor [2x3x3]
    """
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor_grid = torch.zeros((2, 3, 3)).to(DEVICE)
    # tensor_grid[0] must contain positions taken by player
    if player == 'X':
        tensor_grid[0, grid == 1] = 1
        tensor_grid[1, grid == -1] = 1
    if player == 'O':
        tensor_grid[0, grid == -1] = 1
        tensor_grid[1, grid == 1] = 1
    return tensor_grid


def tensor_to_grid(tensor_grid, player='X'):
    """
    Transform a 2x3x3 torch.Tensor to a 3x3 grid
    :param tensor_grid: torch.Tensor [2x3x3]
    :param player: 'X' or 'O'
    :return: np.ndarray [3x3]
    """
    tensor_grid = tensor_grid.to('cpu')
    grid = np.zeros((3,3))
    if player == 'X':
        grid[tensor_grid[0] == 1] = 1
        grid[tensor_grid[1] == 1] = -1
    if player == 'O':
        grid[tensor_grid[0] == 1] = -1
        grid[tensor_grid[1] == 1] = 1
    return grid


def get_other_player(player='X'):
   """
   Get the other player (O if player=X, X if player=O)
   :param player: 'X' or 'O'
    :return: 'O' or 'X'
   """
   if player == 'X':
       return 'O'
   elif player == 'O':
       return 'X'
   else:
       warnings.warn("get_other_player got an unexpected input.")

