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
    return opt


def get_value(pi, q):
    v = dict()
    for s, prob in pi.items():
        s1 = prob[0] * q[(s, 0)]
        s2 = prob[1] * q[(s, 1)]
        v[s] = s1+s2
    return v

class MCExploringStartsAlgorithm:
    @staticmethod
    def train(gamma, epochs):
        q = dict()
        qCounters = dict()
        pi = dict()
        for s in all_states:
            q[(s, 0)] = 0
            q[(s, 1)] = 0
            qCounters[(s, 0)] = 0
            qCounters[(s, 1)] = 0
            pi[s] = random.choice(((1, 0), (0, 1)))

        for i in range(epochs):
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
                a = HIT if q[(s, HIT)] > q[(s, STICK)] else STICK
                pi[s] = (1-a, a)

        return pi, q, get_value(pi, q)

class MCEpsiSoftAlgorithm:
    @staticmethod
    def train(gamma, eps, epochs):
        q = dict()
        qCounters = dict()
        pi = dict()
        for s in all_states:
            q[(s, 0)] = 0
            q[(s, 1)] = 0
            qCounters[(s, 0)] = 0
            qCounters[(s, 1)] = 0
            pi[s] = random.choice(((1-eps, eps), (eps, 1-eps)))

        for i in range(epochs):
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
                prob = [0, 0]
                if q[(s, HIT)] > q[(s, STICK)]:
                    prob[HIT] = 1-eps
                    prob[STICK] = eps
                else:
                    prob[HIT] = eps
                    prob[STICK] = 1-eps
                pi[s] = prob

        return pi, q, get_value(pi, q)



