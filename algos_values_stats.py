#!/usr/bin/env python3
from stats_generators import *
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--episodes", type=int, required=True)
ap.add_argument("-t", "--trainings", type=int, required=True)
args = vars(ap.parse_args())

episodes = args["episodes"]
trainings = args["trainings"]
ylabel = "Sum of value function over states"

def insert_opt(stats, labels, Y):
    stats.insert(0, ([20 for i in range(len(Y))]))
    labels.insert(0, ("optimal"))


params = [(1,), (0.9,), (0.5,), (0.2,)]
algo_name = "MCES"
print('{}: evaluating {} sets of params. Result averaged over {} trainings each of {} episodes.'.format(algo_name, len(params), trainings, episodes))
name = "graphs/{}-{}-{}.jpg".format(algo_name, episodes, trainings)
Y, stats, labels = generate_stats(params, episodes, trainings, MCExploringStartsAlgorithm, ValueFunctionStats)
insert_opt(stats, labels, Y)
get_plot(Y, stats, labels, title="MCES (gamma)", ylabel=ylabel).savefig(name)

params = [(1, 0.01), (0.9, 0.01), (0.5, 0.01), (1, 0.1)]
algo_name = "MCSOFT"
print('{}: evaluating {} sets of params. Result averaged over {} trainings each of {} episodes.'.format(algo_name, len(params), trainings, episodes))
name = "graphs/{}-{}-{}.jpg".format(algo_name, episodes, trainings)
Y, stats, labels = generate_stats(params, episodes, trainings, MCEpsiSoftAlgorithm, ValueFunctionStats)
insert_opt(stats, labels, Y)
get_plot(Y, stats, labels, title="MCSOFT (gamma, eps)", ylabel=ylabel).savefig(name)

params = [(1, 0.01, 0.5), (1, 0.01, 0.2), (1, 0.01, 0.1), (1, 0.01, 0.05)]
algo_name = "TDSARSA"
print('{}: evaluating {} sets of params. Result averaged over {} trainings each of {} episodes.'.format(algo_name, len(params), trainings, episodes))
name = "graphs/{}-{}-{}.jpg".format(algo_name, episodes, trainings)
Y, stats, labels = generate_stats(params, episodes, trainings, TDSarsaAlgorithm, ValueFunctionStats)
insert_opt(stats, labels, Y)
get_plot(Y, stats, labels, title="TDSARSA (gamma, eps, alfa)", ylabel=ylabel).savefig(name)

params = [(1, 0.01, 0.5), (1, 0.01, 0.2), (1, 0.01, 0.1), (1, 0.01, 0.05)]
algo_name = "TDQL"
print('{}: evaluating {} sets of params. Result averaged over {} trainings each of {} episodes.'.format(algo_name, len(params), trainings, episodes))
name = "graphs/{}-{}-{}.jpg".format(algo_name, episodes, trainings)
Y, stats, labels = generate_stats(params, episodes, trainings, TDSarsaAlgorithm, ValueFunctionStats)
insert_opt(stats, labels, Y)
get_plot(Y, stats, labels, title="TDQL (gamma, eps, alfa)", ylabel=ylabel).savefig(name)