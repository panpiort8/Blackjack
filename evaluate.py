#!/usr/bin/env python3
from src.algorithms import *
import argparse

def evaluate(games, pi):
    wins = 0; draws = 0; looses = 0; sum=0
    for i in range(games):
        r = play_game(pi)[-1][2]
        sum += r
        if r == 1:
            wins += 1
        elif r == 0:
            draws += 1
        elif r == -1:
            looses += 1
    return wins, draws, looses, sum

def evaluate_and_print(name, games, pi):
    wins, draws, looses, sum = evaluate(games, pi)
    l = 10
    while len(name) < l:
        name += " "
    print("{4} wins: {0:.0f}%    draws: {1:.0f}%    looses: {2:.0f}%    avg reward: {3:.3f}".
          format(100*wins/games, 100*draws/games, 100*looses/games, sum/games, name))


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--train_episodes", type=int, required=True)
ap.add_argument("-v", "--eval_episodes", type=int, default=10000)
args = vars(ap.parse_args())

# training part
train_episodes = args["train_episodes"]
eval_episodes = args["eval_episodes"]

mces = MCExploringStartsAlgorithm(gamma=0.5)
mcsoft = MCEpsiSoftAlgorithm(gamma=1, eps=0.01)
tdsar = TDSarsaAlgorithm(gamma=1, eps=0.01, alfa=0.2)
tdql = TDQlearningAlgorithm(gamma=1, eps=0.01, alfa=0.1)

widgets = ["MCES training:    ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=train_episodes, widgets=widgets).start()
mces.train(train_episodes, pbar=pbar)
pbar.finish()

widgets = ["MCSOFT training:  ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=train_episodes, widgets=widgets).start()
mcsoft.train(train_episodes, pbar=pbar)
pbar.finish()

widgets = ["TDSARSA training: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=train_episodes, widgets=widgets).start()
tdsar.train(train_episodes, pbar=pbar)
pbar.finish()

widgets = ["TDQL training:    ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=train_episodes, widgets=widgets).start()
tdql.train(train_episodes, pbar=pbar)
pbar.finish()

pi_opt, q_opt, v_opt = load_optimal()
evaluate_and_print("OPTIMAL", eval_episodes, pi_opt)
evaluate_and_print("MCES", eval_episodes, mces.get_pi())
evaluate_and_print("MCSOFT", eval_episodes, mcsoft.get_pi())
evaluate_and_print("TDSARSA", eval_episodes, tdsar.get_pi())
evaluate_and_print("TDQL", eval_episodes, tdql.get_pi())

