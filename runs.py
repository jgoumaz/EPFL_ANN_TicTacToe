import time
from utils import *
from Players import *
from tic_env import TictactoeEnv, OptimalPlayer


def test_against_Opt(Player, n_games=250, opt_eps=0.0, Player_player='X'):
    wins = 0
    losses = 0
    Tictactoe = TictactoeEnv()
    Opt = OptimalPlayer(epsilon=opt_eps)
    Opt_player = get_other_player(Player_player)
    for n in range(n_games):
        Tictactoe.reset()
        for turn in range(9):
            old_grid, _, _ = Tictactoe.observe()
            if Tictactoe.get_current_player() == Opt_player:
                move = Opt.act(old_grid)
            else:
                move = Player.act(old_grid)
            new_grid, end, winner = Tictactoe.step(move)
            if end:
                if winner == Player_player:
                    wins += 1
                elif winner == Opt_player:
                    losses += 1
                break
    return wins, losses

def compute_M_opt(Player):
    wins1, losses1 = test_against_Opt(Player, n_games=250, opt_eps=0.0, Player_player='X')
    wins2, losses2 = test_against_Opt(Player, n_games=250, opt_eps=0.0, Player_player='O')
    wins = wins1 + wins2
    losses = losses1 + losses2
    M_opt = (wins - losses) / 500
    return M_opt

def compute_M_rand(Player):
    wins1, losses1 = test_against_Opt(Player, n_games=250, opt_eps=1.0, Player_player='X')
    wins2, losses2 = test_against_Opt(Player, n_games=250, opt_eps=1.0, Player_player='O')
    wins = wins1 + wins2
    losses = losses1 + losses2
    M_rand = (wins - losses) / 500
    return M_rand


def run_against_Opt(Player, n_games=20000, opt_eps=0.5, return_M_opt=False, return_M_rand=False):
    rewards = []
    average_rewards = []
    M_opts = []
    M_rands = []
    wins = 0
    ties = 0
    losses = 0
    Tictactoe = TictactoeEnv()
    Opt = OptimalPlayer(epsilon=opt_eps)
    for n in range(n_games):
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
    for n in range(n_games):
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


if __name__ == '__main__':
    random.seed(0)
    t0 = time.time()

    Player = QLearningPlayer(eps=0.3, decreasing_exploration=False)
    a, b, c = [], [], []
    a, b = run_against_itself(Player, n_games=1000, return_M_opt=True, return_M_rand=True)
    # a, b, c = run_against_Opt(Player, n_games=1000, return_M_opt=True, return_M_rand=True)
    print(a, b, c)

    t1 = time.time()
    print(f"Total time: {round(t1-t0,2)} sec")

