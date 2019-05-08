from src.algorithms import *

class StatsGenerator:
    def __init__(self, jump = 10):
        self.stats = []
        self.jump = jump

    def add_stat(self, i, pi, q = None):
        pass

    def get_stats(self):
        return self.stats

class CompareOptimalChoices(StatsGenerator):
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