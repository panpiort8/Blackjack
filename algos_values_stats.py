#!/usr/bin/env python3
from stats_generators import *

episodes = 2500
trainings = 2
ylabel = "Sum of value function over states"


params = [(1,), (0.9,), (0.5,), (0.2,)]
algo_name = "MCES"
print('{}: evaluating {} sets of params. Result averaged over {} trainings each of {} episodes.'.format(algo_name, len(params), trainings, episodes))
name = "graphs/{}-{}-{}.jpg".format(algo_name, episodes, trainings)
Y, stats, labels = generate_stats(params, episodes, trainings, MCExploringStartsAlgorithm, ValueFunctionStats)
get_plot(Y, stats, labels, title="MCES (gamma)", ylabel=ylabel).show()

params = [(1, 0.01), (0.9, 0.01), (0.5, 0.01), (1, 0.1)]
algo_name = "MCSOFT"
print('{}: evaluating {} sets of params. Result averaged over {} trainings each of {} episodes.'.format(algo_name, len(params), trainings, episodes))
name = "graphs/{}-{}-{}.jpg".format(algo_name, episodes, trainings)
Y, stats, labels = generate_stats(params, episodes, trainings, MCEpsiSoftAlgorithm, ValueFunctionStats)
get_plot(Y, stats, labels, title="MCSOFT (gamma, eps)", ylabel=ylabel).show()

params = [(1, 0.01, 0.5), (1, 0.01, 0.2), (1, 0.01, 0.1), (1, 0.01, 0.05)]
algo_name = "TDSARSA"
print('{}: evaluating {} sets of params. Result averaged over {} trainings each of {} episodes.'.format(algo_name, len(params), trainings, episodes))
name = "graphs/{}-{}-{}.jpg".format(algo_name, episodes, trainings)
Y, stats, labels = generate_stats(params, episodes, trainings, TDSarsaAlgorithm, ValueFunctionStats)
get_plot(Y, stats, labels, title="TDSARSA (gamma, eps, alfa)", ylabel=ylabel).show()

params = [(1, 0.01, 0.5), (1, 0.01, 0.2), (1, 0.01, 0.1), (1, 0.01, 0.05)]
algo_name = "TDQL"
print('{}: evaluating {} sets of params. Result averaged over {} trainings each of {} episodes.'.format(algo_name, len(params), trainings, episodes))
name = "graphs/{}-{}-{}.jpg".format(algo_name, episodes, trainings)
Y, stats, labels = generate_stats(params, episodes, trainings, TDSarsaAlgorithm, ValueFunctionStats)
get_plot(Y, stats, labels, title="TDQL (gamma, eps, alfa)", ylabel=ylabel).show()