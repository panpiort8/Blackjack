#!/usr/bin/env python3
from src.algorithms import *
import matplotlib.pyplot as plt
import numpy as np

episodes = 25000
trainings = 20
params = [(1, 0.01), (0.9, 0.01), (0.5, 0.01), (1, 0.1)]

stats = [ [] for i in params]
for p, (gamma, eps) in enumerate(params):
    for i in range(trainings):
        print("{}/{}:".format((p * trainings + i + 1), trainings * len(params)))
        pi_es, q_es, v_es, stats_part = MCEpsiSoftAlgorithm.train(gamma, eps, episodes, ValueFunctionStats())
        stats[p].append(stats_part)

plt.style.use('ggplot')
plt.figure()
plt.title('MCSOFT comparison (gamma, eps)')
plt.xlabel('Episodes')
plt.ylabel('Sum of value function over states')
jmp = episodes//len(stats[0][0])
y = [i for i in range(jmp, episodes+1, jmp)]

sum = 20
stats_opt = [sum for i in range(len(y))]

plt.plot(y, stats_opt, label='Optimal')

for i, pars in enumerate(params):
    stat = np.average(np.array(stats[i]), axis=0)
    plt.plot(y, stat, label=str(pars))

plt.legend()
name = "graphs/mcsoft-{}-{}.jpg".format(episodes, trainings)
plt.savefig(name)