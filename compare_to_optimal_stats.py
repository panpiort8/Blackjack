#!/usr/bin/env python3
from stats_generators import *

episodes = 25000
trainings = 1
ylabel = "Choices same as in optimal strategy (%)"
opt, q, v = load_optimal()
stats = []
labels = []

params = [(1,)]
algo_name = "MCES"
print('{}: evaluating similarity to optimal. Result averaged over {} trainings each of {} episodes.'.format(algo_name, trainings, episodes))
Y, stat, _ = generate_stats(params, episodes, trainings, MCExploringStartsAlgorithm, SimilarityToOptimal, args=opt)
stats.append(stat[0])
labels.append(algo_name)

params = [(1, 0.01)]
algo_name = "MCSOFT"
print('{}: evaluating similarity to optimal. Result averaged over {} trainings each of {} episodes.'.format(algo_name, trainings, episodes))
Y, stat, _ = generate_stats(params, episodes, trainings, MCEpsiSoftAlgorithm, SimilarityToOptimal, args=opt)
stats.append(stat[0])
labels.append(algo_name)

params = [(1, 0.01, 0.2)]
algo_name = "TDSARSA"
print('{}: evaluating similarity to optimal. Result averaged over {} trainings each of {} episodes.'.format(algo_name, trainings, episodes))
Y, stat, _ = generate_stats(params, episodes, trainings, TDSarsaAlgorithm, SimilarityToOptimal, args=opt)
stats.append(stat[0])
labels.append(algo_name)

params = [(1, 0.01, 0.2)]
algo_name = "TDQL"
print('{}: evaluating similarity to optimal. Result averaged over {} trainings each of {} episodes.'.format(algo_name, trainings, episodes))
Y, stat, _ = generate_stats(params, episodes, trainings, TDQlearningAlgorithm, SimilarityToOptimal, args=opt)
stats.append(stat[0])
labels.append(algo_name)

get_plot(Y, stats, labels, title="Similarity to optimal stratefy", ylabel=ylabel).show()
