#!/usr/bin/env python3
from src.algorithms import *
import matplotlib.pyplot as plt
import numpy as np

episodes = 25000
trainings = 20
params = [(1), (0.9), (0.5), (0.2)]

stats = [ [] for i in params]
for p, (gamma) in enumerate(params):
    for i in range(trainings):
        print("{}/{}:".format((p*trainings + i + 1), trainings*len(params)))
        pi_es, q_es, v_es, stats_part = MCExploringStartsAlgorithm.train(gamma, episodes, ValueFunctionStats())
        stats[p].append(stats_part)

plt.style.use('ggplot')
plt.figure()
plt.title('MCES comparison (gamma)')
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
name = "graphs/mces-{}-{}.jpg".format(episodes, trainings)
plt.savefig(name)