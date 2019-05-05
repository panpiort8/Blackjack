#!/usr/bin/env python3
from algorithms import *

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
    print("{4} wins: {0:.0f}%    draws: {1:.0f}%    looses: {2:.0f}%    avg: {3:.3f}".
          format(100*wins/games, 100*draws/games, 100*looses/games, sum/games, name))

# training part
train_epochs = 10000
pi_es, q_es, v_es = MCExploringStartsAlgorithm.train(1, train_epochs)
pi_ep, q_ep, v_ep = MCEpsiSoftAlgorithm.train(1, 0.01, train_epochs)
pi_sar, q_sar, v_sar = TDSarsaAlgorithm.train(1, 0.01, 1, train_epochs)
pi_ql, q_ql, v_ql = TDQlearningAlgorithm.train(1, 0.01, 1, train_epochs)

eval_episodes = 10000
pi_opt, q_opt, v_opt = load_optimal()
evaluate_and_print("OPT", eval_episodes, pi_opt)
evaluate_and_print("MCES", eval_episodes, pi_es)
evaluate_and_print("MCSOFT", eval_episodes, pi_ep)
evaluate_and_print("TDSAR", eval_episodes, pi_sar)
evaluate_and_print("TDQL", eval_episodes, pi_ql)
