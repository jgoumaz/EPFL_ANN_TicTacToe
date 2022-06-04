from utils import *
from Players import *
from tic_env import TictactoeEnv, OptimalPlayer


def compute_M_opt(Player):
    pass


def compute_M_rand(Player):
    pass


def run_against_Opt(Player, n_games=100):
    rewards = []
    average_rewards = []
    M_opts = []
    M_rands = []
    wins = 0
    ties = 0
    losses = 0
    Tictactoe = TictactoeEnv()
    Opt = OptimalPlayer()
    for n in range(n_games):
        Tictactoe.reset()
        Opt.set_player(n)
        Player.set_player(n+1)
        for round in range(9):
            if Tictactoe.get_current_player() == Opt.player:
                move = Opt.act()
            else:
                move = Player.act()
            old_grid, _, _ = Tictactoe.observe()
            new_grid, end, winner = Tictactoe.step(move)
            if Tictactoe.get_current_player() == Player.player:
                reward = Tictactoe.reward(player=Player.player)
                Player.update_Q(old_grid, move, new_grid, reward)
            if end:
                if winner == Player.player:
                    wins += 1
                elif winner == Opt.player:
                    losses += 1
                else:
                    ties += 1
                reward = Tictactoe.reward(player=Player.player)
                rewards.append(reward)
        if n%250 == 0:
            average_rewards.append(sum(rewards)/len(rewards))
            rewards = []
            M_opts.append(compute_M_opt(Player))
            M_rands.append(compute_M_rand(Player))
    return average_rewards, M_opts, M_rands
