import time
from tqdm import tqdm # tqdm for loading bars

from tic_env import TictactoeEnv, OptimalPlayer
from Players import *


def test_against_Opt(Player, n_games=250, opt_eps=0.0, Player_player='X'):
    """
    Function used to play multiple games againt Opt for metrics computing purpose.
    :param Player: Player (QLearningPlayer or DeepQLearningPlayer)
    :param n_games: number of games (int)
    :param opt_eps: exploration level of OptimalPlayer
    :param Player_player: 'X' or 'O'
    :return: wins (int), losses (int)
    """
    wins = 0
    losses = 0

    # create environment and OptimalPlayer
    Tictactoe = TictactoeEnv()
    Opt = OptimalPlayer(epsilon=opt_eps)

    # set the players
    Opt_player = get_other_player(Player_player)
    Player.set_player(Player_player)
    Opt.set_player(Opt_player)

    for n in range(n_games):
        Tictactoe.reset()
        for turn in range(9): # one game with a maximal number of 9 iterations
            old_grid, _, _ = Tictactoe.observe()
            if Tictactoe.get_current_player() == Opt_player:
                move = Opt.act(old_grid)
                move_valid = True
            else:
                move = Player.act(old_grid)
                move_valid = Tictactoe.check_valid(move)

            if move_valid:
                new_grid, end, winner = Tictactoe.step(move)
            else: # illegal move
                end = True
                winner = Opt_player
            if end: # count wins/losses
                if winner == Player_player:
                    wins += 1
                elif winner == Opt_player:
                    losses += 1
                break
    return wins, losses


def compute_M_opt(Player):
    """
    Computes M_opt with 500 games
    :param Player: Player (QLearningPlayer or DeepQLearningPlayer)
    :return: M_opt (float)
    """
    wins1, losses1 = test_against_Opt(Player, n_games=250, opt_eps=0.0, Player_player='X')
    wins2, losses2 = test_against_Opt(Player, n_games=250, opt_eps=0.0, Player_player='O')
    wins = wins1 + wins2
    losses = losses1 + losses2
    M_opt = (wins - losses) / 500
    return M_opt


def compute_M_rand(Player):
    """
    Computes M_rand with 500 games
    :param Player: Player (QLearningPlayer or DeepQLearningPlayer)
    :return: M_rand (float)
    """
    wins1, losses1 = test_against_Opt(Player, n_games=250, opt_eps=1.0, Player_player='X')
    wins2, losses2 = test_against_Opt(Player, n_games=250, opt_eps=1.0, Player_player='O')
    wins = wins1 + wins2
    losses = losses1 + losses2
    M_rand = (wins - losses) / 500
    return M_rand


def run_against_Opt(Player, n_games=20000, opt_eps=0.5, return_M_opt=False, return_M_rand=False):
    """
    Run games with Player against OptimalPlayer. The metrics are evaluated each 250 games.
    :param Player: Player (QLearningPlayer or DeepQLearningPlayer)
    :param n_games: number of games (int)
    :param opt_eps: exploration level of OptimalPlayer
    :param return_M_opt: computes and returns M_opt if True else M_opts=[]
    :param return_M_rand: computes and returns M_rand if True else M_rands=[]
    :return: average_rewards (list), M_opts (list), M_rands (list)
    """
    rewards = []
    average_rewards = []
    M_opts = []
    M_rands = []
    wins = 0
    ties = 0
    losses = 0
    Tictactoe = TictactoeEnv()
    Opt = OptimalPlayer(epsilon=opt_eps)
    for n in tqdm(range(n_games)): # tqdm for loading bars
        Tictactoe.reset()
        old_grids = []
        moves = []
        Opt.set_player(j=n)
        Player.set_player(j=n+1)
        for turn in range(9):
            old_grid, _, _ = Tictactoe.observe()
            old_grids.append(old_grid)
            if Tictactoe.get_current_player() == Opt.player:
                move = Opt.act(old_grid)
            else:
                move = Player.act(old_grid)
            moves.append(move)
            new_grid, end, winner = Tictactoe.step(move)
            if (Tictactoe.get_current_player() == Player.player and turn > 0) or end:
                reward = Tictactoe.reward(player=Player.player)
                if reward == 1: # The case when Player wins
                    Player.update_Q(new_grid, reward, old_grids[-1], moves[-1])
                else: # All other cases
                    Player.update_Q(new_grid, reward, old_grids[-2], moves[-2])
            if end:
                if winner == Player.player:
                    wins += 1
                elif winner == Opt.player:
                    losses += 1
                else:
                    ties += 1
                reward = Tictactoe.reward(player=Player.player)
                rewards.append(reward)
                break
        Player.n += 1
        if n%250 == 249:
            average_rewards.append(sum(rewards)/len(rewards))
            rewards = []
            Player.best_play = True
            if return_M_opt: M_opts.append(compute_M_opt(Player))
            if return_M_rand: M_rands.append(compute_M_rand(Player))
            Player.best_play = False
    return average_rewards, M_opts, M_rands

def run_against_itself(Player, n_games=20000, return_M_opt=False, return_M_rand=False):
    M_opts = []
    M_rands = []
    Tictactoe = TictactoeEnv()
    for n in tqdm(range(n_games)): # tqdm for loading bars
        Tictactoe.reset()
        old_grids = []
        moves = []
        for turn in range(9):
            old_grid, _, _ = Tictactoe.observe()
            old_grids.append(old_grid)
            move = Player.act(old_grid)
            moves.append(move)
            new_grid, end, winner = Tictactoe.step(move)
            if turn > 0:
                reward = Tictactoe.reward(player=Tictactoe.get_current_player())
                Player.update_Q(new_grid, reward, old_grids[-2], moves[-2])
            if end:
                reward = Tictactoe.reward(player=winner)
                Player.update_Q(new_grid, reward, old_grids[-1], moves[-1])
                break
        Player.n += 1
        if n%250 == 249:
            Player.best_play = True
            if return_M_opt: M_opts.append(compute_M_opt(Player))
            if return_M_rand: M_rands.append(compute_M_rand(Player))
            Player.best_play = False
    return M_opts, M_rands

def run_DQN_against_Opt(Player, n_games=20000, opt_eps=0.5, return_M_opt=False, return_M_rand=False, return_average_loss=False):
    rewards = []
    average_rewards = []
    average_loss = []
    Player.save_loss = return_average_loss
    M_opts = []
    M_rands = []
    wins = 0
    ties = 0
    losses = 0
    Tictactoe = TictactoeEnv()
    Opt = OptimalPlayer(epsilon=opt_eps)
    for n in tqdm(range(n_games)): # tqdm for loading bars
        Tictactoe.reset()
        old_grids = []
        moves = []
        Opt.set_player(j=n)
        Player.set_player(j=n+1)

        for turn in range(9):
            old_grid, _, _ = Tictactoe.observe()
            old_grids.append(grid_to_tensor(old_grid, player=Player.player))
            if Tictactoe.get_current_player() == Opt.player:
                move = position_to_index(Opt.act(old_grid))
                move_valid = True
            else:
                move = Player.act(old_grid)
                move_valid = Tictactoe.check_valid(move)
            moves.append(move)
            if move_valid:
                new_grid, end, winner = Tictactoe.step(move)
                new_grid = grid_to_tensor(new_grid, player=Player.player)
                reward = Tictactoe.reward(player=Player.player)
            else:
                end = True
                winner = Opt.player
                reward = -1
            if end:
                new_grid = None
            if (Tictactoe.get_current_player() == Player.player and turn > 0) or end:
                if reward == 1 or move_valid == False: # The case when Player wins or makes illegal move
                    Player.buffer.store(old_grids[-1], moves[-1], new_grid, reward)
                else: # All other cases
                    Player.buffer.store(old_grids[-2], moves[-2], new_grid, reward)
                Player.optimize_model()
            if end:
                if winner == Player.player:
                    wins += 1
                elif winner == Opt.player:
                    losses += 1
                else:
                    ties += 1
                rewards.append(reward)
                break

        Player.n += 1
        if n%500 == 499:
            Player.target_net.load_state_dict(Player.policy_net.state_dict())
        if n%250 == 249:
            average_rewards.append(sum(rewards)/len(rewards))
            rewards = []
            Player.best_play = True
            Player.save_loss = False
            if return_M_opt: M_opts.append(compute_M_opt(Player))
            if return_M_rand: M_rands.append(compute_M_rand(Player))
            if return_average_loss: average_loss.append(Player.get_loss_average())
            Player.best_play = False
            Player.save_loss = return_average_loss
    return average_rewards, M_opts, M_rands, average_loss

def run_DQN_against_itself(Player, n_games=20000, return_M_opt=False, return_M_rand=False, return_average_loss=False):
    average_loss = []
    Player.save_loss = return_average_loss
    M_opts = []
    M_rands = []
    Tictactoe = TictactoeEnv()
    for n in tqdm(range(n_games)): # tqdm for loading bars
        Tictactoe.reset()
        old_grids = []
        moves = []
        for turn in range(9):
            old_grid, _, _ = Tictactoe.observe()
            old_grids.append(grid_to_tensor(old_grid, player=Tictactoe.get_current_player()))
            move = Player.act(old_grids[-1])
            moves.append(move)
            move_valid = Tictactoe.check_valid(move)
            if move_valid:
                new_grid, end, winner = Tictactoe.step(move)
                new_grid = grid_to_tensor(new_grid, player=Tictactoe.get_current_player())
                reward1 = Tictactoe.reward(player=get_other_player(Tictactoe.get_current_player()))
                reward2 = Tictactoe.reward(player=Tictactoe.get_current_player())
            else:
                end = True
                reward1 = -1
                # reward2 = 0 # not needed
            if end:
                new_grid = None
            if turn > 0:
                if move_valid:
                    Player.buffer.store(old_grids[-2], moves[-2], new_grid, reward2)
                    if end:
                        Player.buffer.store(old_grids[-1], moves[-1], new_grid, reward1) # If Player wins (in this iteration)
                else:
                    Player.buffer.store(old_grids[-1], moves[-1], new_grid, reward1) # If Player makes an illegal move (in this iteration)
            Player.optimize_model()
            if end:
                break
        Player.n += 1
        if n%500 == 499:
            Player.target_net.load_state_dict(Player.policy_net.state_dict())
        if n%250 == 249:
            Player.best_play = True
            Player.save_loss = False
            if return_M_opt: M_opts.append(compute_M_opt(Player))
            if return_M_rand: M_rands.append(compute_M_rand(Player))
            if return_average_loss: average_loss.append(Player.get_loss_average())
            Player.best_play = False
            Player.save_loss = return_average_loss
    return M_opts, M_rands, average_loss


if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)
    a, b, c, d = [], [], [], []
    t0 = time.time()

    # Player = QLearningPlayer(eps=0.3, decreasing_exploration=False)
    # a, b, c = run_against_Opt(Player, n_games=1000, return_M_opt=True, return_M_rand=True)
    # a, b = run_against_itself(Player, n_games=1000, return_M_opt=True, return_M_rand=True)
    Player = DeepQLearningPlayer(eps=0.3, decreasing_exploration=True, n_star=100)
    # a, b, c, d = run_DQN_against_Opt(Player, n_games=20000, return_M_opt=True, return_M_rand=True, return_average_loss=True)
    a, b, c = run_DQN_against_itself(Player, n_games=20000, return_M_opt=True, return_M_rand=True, return_average_loss=True)
    print(a, b, c, d)

    t1 = time.time()
    print(f"Total time: {round(t1-t0,2)} sec")

