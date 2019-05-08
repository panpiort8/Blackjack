import progressbar

from src.environment import *

def load_optimal():
    opt = dict()
    for s in all_states:
        hand, dealer_hand, usable = s
        if usable:
            if hand <= 17 or (hand == 18 and dealer_hand in (11, 10, 9)):
                a = HIT
            else:
                a = STICK
        else:
            if (hand <= 16 and dealer_hand in (7, 8, 9, 10, 11)) \
                    or (hand <= 12 and dealer_hand in (2, 3)):
                a = HIT
            else:
                a = STICK
        opt[s] = (1-a, a)
    q = evaluate_q(opt)
    return opt, q, get_value(opt, q)

def get_value(pi, q):
    v = dict()
    for s, prob in pi.items():
        s1 = prob[0] * q[(s, 0)]
        s2 = prob[1] * q[(s, 1)]
        v[s] = s1+s2
    return v

def soften(prob, eps):
    return (1-eps, eps) if prob == (1, 0) else (eps, 1-eps)

def soften_pi(pi, eps):
    for s in all_states:
        pi[s] = soften(pi[s], eps)

def greedy(s, q):
    a = HIT if q[(s, HIT)] > q[(s, STICK)] else STICK
    return (1-a, a)

def greedy_pi(pi, q):
    for s in all_states:
        pi[s] = greedy(s, q)

def get_nontrivial_start_state():
    while True:
        start = get_start_state()
        if start[1] not in ("tie", "natural"):
            return start



class Algorithm:
    def initialize(self):
        self.q = dict()
        self.pi = dict()
        for s in all_states:
            self.q[(s, 0)] = np.random.normal(0, 0.01)
            self.q[(s, 1)] = np.random.normal(0, 0.01)
            self.pi[s] = greedy(s, self.q)

    def step(self):
        pass

    def finalize(self):
        pass

    def train(self, n_episodes, stats_maker = None, pbar = None):
        self.initialize()
        for i in range(n_episodes):
            self.step()
            if stats_maker is not None:
                stats_maker.add_stat(i, self.pi, self.q)
            if pbar is not None:
                pbar.update(i)
        self.finalize()
        if stats_maker is not None:
            return stats_maker.get_stats()

    def get_pi(self):
        return self.pi

    def get_q(self):
        return self.q

    def get_v(self):
        return get_value(self.pi, self.q)


class MCExploringStartsAlgorithm(Algorithm):
    def __init__(self, gamma, **kwargs):
        self.gamma = gamma

    def initialize(self):
        super(MCExploringStartsAlgorithm, self).initialize()
        self.qCounters = dict()
        for s in all_states:
            self.qCounters[(s, 0)] = self.qCounters[(s, 1)] = 0

    def step(self):
        episode = generate_episode(random.choice(all_states), self.pi)
        rewards = dict()
        for s, a, _ in episode:
            rewards[(s, a)] = 0

        G = 0
        for s, a, r in reversed(episode):
            G = self.gamma * G + r
            rewards[(s, a)] = G

        for sa, r in rewards.items():
            self.qCounters[sa] += 1
            self.q[sa] = self.q[sa] + 1 / self.qCounters[sa] * (r - self.q[sa])
            s, a = sa
            self.pi[s] = greedy(s, self.q)


class QEvaluateAlgorithm(MCExploringStartsAlgorithm):
    def __init__(self, pi, **kwargs):
        super().__init__(1)
        self.constpi = pi

    def initialize(self):
        super().initialize()
        self.pi = self.constpi

    def step(self):
        episode = generate_episode(random.choice(all_states), self.pi)
        rewards = dict()
        for s, a, _ in episode:
            rewards[(s, a)] = 0

        G = 0
        for s, a, r in reversed(episode):
            G = self.gamma * G + r
            rewards[(s, a)] = G

        for sa, r in rewards.items():
            self.qCounters[sa] += 1
            self.q[sa] = self.q[sa] + 1 / self.qCounters[sa] * (r - self.q[sa])

def evaluate_q(pi):
    e = QEvaluateAlgorithm(pi)
    e.train(10000)
    return e.get_q()


class MCEpsiSoftAlgorithm(MCExploringStartsAlgorithm):
    def __init__(self, gamma, eps, **kwargs):
        super().__init__(gamma)
        self.eps = eps

    def initialize(self):
        super().initialize()
        soften_pi(self.pi, self.eps)

    def step(self):
        episode = generate_episode(get_nontrivial_start_state(), self.pi)
        rewards = dict()
        for s, a, _ in episode:
            rewards[(s, a)] = 0

        G = 0
        for s, a, r in reversed(episode):
            G = self.gamma * G + r
            rewards[(s, a)] = G

        for sa, r in rewards.items():
            self.qCounters[sa] += 1
            self.q[sa] = self.q[sa] + 1 / self.qCounters[sa] * (r - self.q[sa])
            s, _ = sa
            self. pi[s] = soften(greedy(s, self.q), self.eps)

    def finalize(self):
        greedy_pi(self.pi, self.q)


class TDSarsaAlgorithm(Algorithm):
    def __init__(self, gamma, eps, alfa, **kwargs):
        self.gamma = gamma
        self.eps = eps
        self.alfa = alfa

    def initialize(self):
        super().initialize()
        for s in [(-1, 'end', -1), (-1, 'burst', -1)]:
            self.pi[s] = (0, 1)
            for a in (0, 1):
                self.q[(s, a)] = 0

    def step(self):
        s = get_nontrivial_start_state()
        a = np.random.choice((0, 1), p=self.pi[s])
        while s[1] not in ('end', 'burst'):
            a, r, ns = take_determined_action(s, a)
            na = np.random.choice((0, 1), p=self.pi[ns])
            self.q[(s, a)] += self.alfa * (r + self.gamma * self.q[(ns, na)] - self.q[(s, a)])
            self.pi[s] = soften(greedy(s, self.q), self.eps)
            s = ns
            a = na

    def finalize(self):
        greedy_pi(self.pi, self.q)


class TDQlearningAlgorithm(TDSarsaAlgorithm):
    def step(self):
        s = get_nontrivial_start_state()
        while s[1] not in ('end', 'burst'):
            a, r, ns = take_action(s, self.pi)
            self.q[(s, a)] += self.alfa*(r + self.gamma*max(self.q[(ns, STICK)], self.q[(ns, HIT)]) - self.q[(s, a)])
            self.pi[s] = soften(greedy(s, self.q), self.eps)
            s = ns

