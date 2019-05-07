#!/usr/bin/env python3
from algorithms import *
import matplotlib.pyplot as plt
import numpy as np

train_epochs = 1000
trainings = 1

opt,*_ = load_optimal()

stats_es = []
for i in range(trainings):
    *_, stats = MCExploringStartsAlgorithm.train(1, train_epochs, CompareOptimalChoices(opt))
    stats_es.append(stats)
stats_es = np.average(np.array(stats_es), axis=0)

stats_ep = []
for i in range(trainings):
    *_, stats = MCEpsiSoftAlgorithm.train(1, 0.01, train_epochs, CompareOptimalChoices(opt))
    stats_ep.append(stats)
stats_ep = np.average(np.array(stats_ep), axis=0)

stats_sar = []
for i in range(trainings):
    *_, stats = TDSarsaAlgorithm.train(1, 0.01, 1, train_epochs, CompareOptimalChoices(opt))
    stats_sar.append(stats)
stats_sar = np.average(np.array(stats_sar), axis=0)

stats_ql = []
for i in range(trainings):
    *_, stats = TDQlearningAlgorithm.train(1, 0.01, 1, train_epochs, CompareOptimalChoices(opt))
    stats_ql.append(stats)
stats_ql = np.average(np.array(stats_ql), axis=0)

jmp = train_epochs//len(stats_es)
y = [i for i in range(jmp, train_epochs+1, jmp)]

plt.style.use('ggplot')
plt.figure()
plt.title('Compare to optimal')
plt.plot(y, stats_es, label='Exploring Starts')
plt.plot(y, stats_ep, label='Epsilon Soft')
plt.plot(y, stats_sar, label='Sarsa')
plt.plot(y, stats_ql, label='Q-learning')
plt.xlabel('Episodes')
plt.ylabel('Optimal choices (%)')
plt.legend()
plt.show()

