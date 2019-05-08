from src import *
import copy
import matplotlib.pyplot as plt

class StatsGenerator:
    def __init__(self, jump = 10):
        self.stats = []
        self.jump = jump

    def add_stat(self, i, pi, q = None):
        pass

    def get_stats(self):
        return self.stats

class SimilarityToOptimal(StatsGenerator):
    def __init__(self, opt, jump = 10):
        super().__init__(jump)
        self.n_states = len(all_states)
        self.opt = opt

    def add_stat(self, i, pi, q = None):
        if (i+1) % self.jump != 0:
            return
        same = 0
        for s in all_states:
            if greedy(s, q) == self.opt[s]:
                same += 1
        self.stats.append(100*same/self.n_states)

class ValueFunctionStats(StatsGenerator):
    def add_stat(self, i, pi, q = None):
        if (i+1) % self.jump != 0:
            return
        v = get_value(pi, q)
        sum = 0
        for s, val in v.items():
            sum += val
        self.stats.append(sum)

def get_plot(Y, Xs, labels, title ="Graph", xlabel="Episodes", ylabel="Some value"):
    plt.style.use('ggplot')
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for X, label in zip(Xs, labels):
        plt.plot(Y, X, label=label)
    plt.legend()
    return plt

def generate_stats(params, episodes, trainings, algoClass, statsClass, args = None):
    stats = [[] for p in params]
    fparams = [list(par) for par in params]
    for p, fparam in enumerate(fparams):
        while len(fparam) < 3:
            fparam.append(0)
        for i in range(trainings):
            header = "{}/{}: ".format((p * trainings + i), trainings * len(params))
            widgets = [header, progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
            pbar = progressbar.ProgressBar(maxval=episodes, widgets=widgets).start()
            algo = algoClass(gamma=fparam[0], eps=fparam[1], alfa=fparam[2])
            if args is not None:
                stats_generator = statsClass(args)
            else:
                stats_generator = statsClass()
            stats[p].append(algo.train(episodes, stats_generator, pbar=pbar))
    pbar.finish()

    jmp = episodes // len(stats[0][0])
    Y = [i for i in range(jmp, episodes + 1, jmp)]
    labels = []
    for i, par in enumerate(params):
        stats[i] = np.average(np.array(stats[i]), axis=0)
        labels.append(str(par))
    sum = 20
    stats.append([sum for i in range(len(Y))])
    labels.append('optimal')

    return Y, stats, labels