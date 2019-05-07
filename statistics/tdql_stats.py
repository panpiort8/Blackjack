#!/usr/bin/env python3
from algorithms import *
import matplotlib.pyplot as plt
import numpy as np

episodes = 1000
trainings = 2
params = [(1, 0.01, 0.5), (0.5, 0.1, 0.01)]


stats = [ [] for i in params]
for p, (gamma, eps, alfa) in enumerate(params):
    for i in range(trainings):
        pi_es, q_es, v_es, stats_part = TDQlearningAlgorithm.train(gamma, eps, alfa, episodes, ValueFunctionStats())
        stats[p].append(stats_part)

plt.style.use('ggplot')
plt.figure()
plt.title('TDQL comparison (gamma, eps, alfa)')
plt.xlabel('Episodes')
plt.ylabel('Sum of value function')
jmp = episodes//len(stats[0][0])
y = [i for i in range(jmp, episodes+1, jmp)]

sum = 20
stats_opt = [sum for i in range(len(y))]

plt.plot(y, stats_opt, label='Optimal')

for i, pars in enumerate(params):
    stat = np.average(np.array(stats[i]), axis=0)
    plt.plot(y, stat, label=str(pars))

plt.legend()
plt.show()