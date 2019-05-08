#!/usr/bin/env python3
from stats_generators import *
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--episodes", type=int, required=True)
ap.add_argument("-t", "--trainings", type=int, required=True)
args = vars(ap.parse_args())

episodes = args["episodes"]
trainings = args["trainings"]
ylabel = "Choices same as in optimal strategy (%)"
opt, q, v = load_optimal()
stats = []
labels = []

params = [(1,)]
algo_name = "MCES {}".format(params[0])
print('{}: evaluating similarity to optimal. Result averaged over {} trainings each of {} episodes.'.format(algo_name, trainings, episodes))
Y, stat, _ = generate_stats(params, episodes, trainings, MCExploringStartsAlgorithm, SimilarityToOptimal, args=opt)
stats.append(stat[0])
labels.append(algo_name)

params = [(1, 0.01)]
algo_name = "MSOFT {}".format(params[0])
print('{}: evaluating similarity to optimal. Result averaged over {} trainings each of {} episodes.'.format(algo_name, trainings, episodes))
Y, stat, _ = generate_stats(params, episodes, trainings, MCEpsiSoftAlgorithm, SimilarityToOptimal, args=opt)
stats.append(stat[0])
labels.append(algo_name)

params = [(1, 0.01, 0.2)]
algo_name = "TDSARSA {}".format(params[0])
print('{}: evaluating similarity to optimal. Result averaged over {} trainings each of {} episodes.'.format(algo_name, trainings, episodes))
Y, stat, _ = generate_stats(params, episodes, trainings, TDSarsaAlgorithm, SimilarityToOptimal, args=opt)
stats.append(stat[0])
labels.append(algo_name)

params = [(1, 0.01, 0.2)]
algo_name = "TDQL {}".format(params[0])
print('{}: evaluating similarity to optimal. Result averaged over {} trainings each of {} episodes.'.format(algo_name, trainings, episodes))
Y, stat, _ = generate_stats(params, episodes, trainings, TDQlearningAlgorithm, SimilarityToOptimal, args=opt)
stats.append(stat[0])
labels.append(algo_name)

name = "graphs/similarity-{}-{}.jpg".format(episodes, trainings)
get_plot(Y, stats, labels, title="Similarity to optimal stratefy", ylabel=ylabel).savefig(name)
