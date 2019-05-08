#!/usr/bin/env python3
from src.algorithms import *
import matplotlib.pyplot as plt
import numpy as np

train_epochs = 10000
trainings = 20

stats_es = []
for i in range(trainings):
    *_, stats = MCExploringStartsAlgorithm.train(1, train_epochs, ValueFunctionStats())
    stats_es.append(stats)
stats_es = np.average(np.array(stats_es), axis=0)

stats_ep = []
for i in range(trainings):
    *_, stats = MCEpsiSoftAlgorithm.train(1, 0.01, train_epochs, ValueFunctionStats())
    stats_ep.append(stats)
stats_ep = np.average(np.array(stats_ep), axis=0)

stats_sar = []
for i in range(trainings):
    *_, stats = TDSarsaAlgorithm.train(1, 0.01, 1, train_epochs, ValueFunctionStats())
    stats_sar.append(stats)
stats_sar = np.average(np.array(stats_sar), axis=0)

stats_ql = []
for i in range(trainings):
    *_, stats = TDQlearningAlgorithm.train(1, 0.01, 1, train_epochs, ValueFunctionStats())
    stats_ql.append(stats)
stats_ql = np.average(np.array(stats_ql), axis=0)

jmp = train_epochs//len(stats_es)
y = [i for i in range(jmp, train_epochs+1, jmp)]
opt, q, opt_val = load_optimal()
sum = 0
for s, val in opt_val.items():
    sum += val
stats_opt = [sum for i in range(len(y))]


plt.style.use('ggplot')
plt.figure()
plt.title('Compare to optimal')
plt.plot(y, stats_opt, label='Optimal')
plt.plot(y, stats_es, label='Exploring Starts')
plt.plot(y, stats_ep, label='Epsilon Soft')
plt.plot(y, stats_sar, label='Sarsa')
plt.plot(y, stats_ql, label='Q-learning')
plt.xlabel('Episodes')
plt.ylabel('Sum of value function')
plt.legend()
plt.show()

