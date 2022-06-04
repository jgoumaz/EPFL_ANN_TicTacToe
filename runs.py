import time
from utils import *
from Players import *
from tic_env import TictactoeEnv, OptimalPlayer


def compute_M_opt(Player):
    wins = 0
    losses = 0
    n_games = 500
    Tictactoe = TictactoeEnv()
    Opt = OptimalPlayer(epsilon=0)
    # 250 games for Opt(0)
    for n in range(n_games/2):
        Tictactoe.reset()
        Opt.set_player(j=n)
        Player.set_player(j=n+1)
        for turn in range(9):
            old_grid, _, _ = Tictactoe.observe()
            if Tictactoe.get_current_player() == Opt.player:
                move = Opt.act(old_grid)
            else:
                move = Player.act(old_grid)
            new_grid, end, winner = Tictactoe.step(move)
            if end:
                if winner == Player.player:
                    wins += 1
                elif winner == Opt.player:
                    losses += 1
    # 250 games for Opt(1)
    for n in range(n_games/2):
        Tictactoe.reset()
        Opt.set_player(j=n+1)
        Player.set_player(j=n)
        for turn in range(9):
            old_grid, _, _ = Tictactoe.observe()
            if Tictactoe.get_current_player() == Opt.player:
                move = Opt.act(old_grid)
            else:
                move = Player.act(old_grid)
            new_grid, end, winner = Tictactoe.step(move)

            if end:
                if winner == Player.player:
                    wins += 1
                elif winner == Opt.player:
                    losses += 1
    M_opt = wins - losses / n_games
    return M_opt


def compute_M_rand(Player):
    wins = 0
    losses = 0
    Tictactoe = TictactoeEnv()
    n_games = 500
    Opt = OptimalPlayer(epsilon=1)
    # 250 games for Opt(0)
    for n in range(n_games / 2):
        Tictactoe.reset()
        Opt.set_player(j=n)
        Player.set_player(j=n+1)
        for turn in range(9):
            old_grid, _, _ = Tictactoe.observe()
            if Tictactoe.get_current_player() == Opt.player:
                move = Opt.act(old_grid)
            else:
                move = Player.act(old_grid)
            new_grid, end, winner = Tictactoe.step(move)
            if end:
                if winner == Player.player:
                    wins += 1
                elif winner == Opt.player:
                    losses += 1
                break
    # 250 games for Opt(1)
    for n in range(n_games / 2):
        Tictactoe.reset()
        Opt.set_player(j=n+1)
        Player.set_player(j=n)
        for turn in range(9):
            old_grid, _, _ = Tictactoe.observe()
            if Tictactoe.get_current_player() == Opt.player:
                move = Opt.act(old_grid)
            else:
                move = Player.act(old_grid)
            new_grid, end, winner = Tictactoe.step(move)
            if end:
                if winner == Player.player:
                    wins += 1
                elif winner == Opt.player:
                    losses += 1
                break
    M_rand = wins - losses / n_games
    return M_rand


def run_against_Opt(Player, n_games=100, opt_eps=0.2, return_M_opt=False, return_M_rand=False):
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
        Opt.set_player(j=n)
        Player.set_player(j=n+1)
        for turn in range(9):
            old_grid, _, _ = Tictactoe.observe()
            if Tictactoe.get_current_player() == Opt.player:
                move = Opt.act(old_grid)
            else:
                move = Player.act(old_grid)
            new_grid, end, winner = Tictactoe.step(move)
            if (Tictactoe.get_current_player() == Player.player and turn > 0) or end:
                reward = Tictactoe.reward(player=Player.player)
                Player.update_Q(new_grid, reward)
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
            if return_M_opt: M_opts.append(compute_M_opt(Player))
            if return_M_rand: M_rands.append(compute_M_rand(Player))
    return average_rewards, M_opts, M_rands


if __name__ == '__main__':
    Player = QLearningPlayer(eps=0.0, decreasing_exploration=True)
    a,b,c = run_against_Opt(Player, n_games=20000, opt_eps=0.5)
    print(a)