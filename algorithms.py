import random
from environment import *

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
    q = evaluate_q(opt, 1, 1000)
    return opt, q, get_value(opt, q)

def get_value(pi, q):
    v = dict()
    for s, prob in pi.items():
        s1 = prob[0] * q[(s, 0)]
        s2 = prob[1] * q[(s, 1)]
        v[s] = s1+s2
    return v

def evaluate_q(pi, gamma, n_episodes):
    q, _ = Algorithm.initialize()
    alfa = 1
    for s in [(-1, 'end', -1), (-1, 'burst', -1)]:
        pi[s] = (0, 1)
        for a in (0, 1):
            q[(s, a)] = 0

    for i in range(n_episodes):
        s = random.choice(all_states)
        while s[1] not in ('end', 'burst'):
            a, r, ns = take_action(s, pi)
            q[(s, a)] += alfa * (r + gamma * max(q[(ns, STICK)], q[(ns, HIT)]) - q[(s, a)])
            s = ns
    return q


def soften(prob, eps):
    return (1-eps, eps) if prob == (1, 0) else (eps, 1-eps)

def soften_pi(pi, eps):
    for s in all_states:
        pi[s] = soften(pi[s], eps)
    return pi

def greedy(s, q):
    a = HIT if q[(s, HIT)] > q[(s, STICK)] else STICK
    return (1-a, a)

class Algorithm:
    @staticmethod
    def initialize():
        q = dict()
        pi = dict()
        for s in all_states:
            q[(s, 0)] = np.random.normal(0, 0.1)
            q[(s, 1)] = np.random.normal(0, 0.1)
            pi[s] = greedy(s, q)
        return q, pi

class MCExploringStartsAlgorithm(Algorithm):
    @staticmethod
    def train(gamma, n_episodes):
        q, pi = Algorithm.initialize()
        qCounters = dict()
        for s in all_states:
            qCounters[(s, 0)] = qCounters[(s, 1)] = 0

        for i in range(n_episodes):
            episode = generate_episode(random.choice(all_states), pi)
            rewards = dict()
            for s, a, _ in episode:
                rewards[(s, a)] = 0

            G = 0
            for s, a, r in reversed(episode):
                G = gamma * G + r
                rewards[(s, a)] = G

            for sa, r in rewards.items():
                qCounters[sa] += 1
                q[sa] = q[sa] + 1 / qCounters[sa] * (r - q[sa])
                s, a = sa
                pi[s] = greedy(s, q)

        return pi, q, get_value(pi, q)

class MCEpsiSoftAlgorithm:
    @staticmethod
    def train(gamma, eps, n_episodes):
        q, pi = Algorithm.initialize()
        pi = soften_pi(pi, eps)
        qCounters = dict()
        for s in all_states:
            qCounters[(s, 0)] = qCounters[(s, 1)] = 0


        for i in range(n_episodes):
            while True:
                start = get_start_state()
                if start[1] not in ("tie", "natural"):
                    break
            episode = generate_episode(start, pi)
            rewards = dict()
            for s, a, _ in episode:
                rewards[(s, a)] = 0

            G = 0
            for s, a, r in reversed(episode):
                G = gamma * G + r
                rewards[(s, a)] = G

            for sa, r in rewards.items():
                qCounters[sa] += 1
                q[sa] = q[sa] + 1 / qCounters[sa] * (r - q[sa])
                s, _ = sa
                pi[s] = soften(greedy(s, q), eps)

        return pi, q, get_value(pi, q)

class TDSarsaAlgorithm:
    @staticmethod
    def train(gamma, eps, alfa, n_episodes):
        q, pi = Algorithm.initialize()
        pi = soften_pi(pi, eps)

        for s in [(-1, 'end', -1), (-1, 'burst', -1)]:
            pi[s] = (0, 1)
            for a in (0, 1):
                q[(s, a)] = 0

        for i in range(n_episodes):
            while True:
                s = get_start_state()
                if s[1] not in ("tie", "natural"):
                    break
            a = np.random.choice((0, 1), p=pi[s])
            while s[1] not in ('end', 'burst'):
                a, r, ns = take_determined_action(s, a)
                na = np.random.choice((0, 1), p=pi[ns])
                q[(s, a)] += alfa*(r + gamma*q[(ns, na)] - q[(s, a)])
                pi[s] = soften(greedy(s, q), eps)
                s = ns; a = na

        return pi, q, get_value(pi, q)

class TDQlearningAlgorithm:
    @staticmethod
    def train(gamma, eps, alfa, n_episodes):
        q, pi = Algorithm.initialize()
        pi = soften_pi(pi, eps)

        for s in [(-1, 'end', -1), (-1, 'burst', -1)]:
            pi[s] = (0, 1)
            for a in (0, 1):
                q[(s, a)] = 0

        for i in range(n_episodes):
            while True:
                s = get_start_state()
                if s[1] not in ("tie", "natural"):
                    break
            while s[1] not in ('end', 'burst'):
                a, r, ns = take_action(s, pi)
                q[(s, a)] += alfa*(r + gamma*max(q[(ns, STICK)], q[(ns, HIT)]) - q[(s, a)])
                pi[s] = soften(greedy(s, q), eps)
                s = ns

        return pi, q, get_value(pi, q)

