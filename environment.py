import random
import numpy as np

cards = [11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
HIT = 0
STICK = 1
all_states = [(h, d, u) for h in range(12, 22) for d in range(2, 12) for u in (True, False)]

def get_card():
    return random.choice(cards)

def process_card(hand, ace_in_use, card):
    if card == 11 and 11 + hand <= 21:
        hand += 11
        ace_in_use = True
        return hand, ace_in_use

    if card == 11:
        hand += 1
    else:
        hand += card

    # need to use ace
    if hand > 21 and ace_in_use:
        hand -= 10
        ace_in_use = False

    return hand, ace_in_use


# returns dealer_hand
def simulate_dealer(dealer_hand):
    dealer_ace_in_use = dealer_hand == 11

    while dealer_hand <= 17:
        dealer_hand, dealer_ace_in_use = process_card(dealer_hand, dealer_ace_in_use, get_card())

    return dealer_hand

def hit(s):
    hand, dealer, ace_in_use = s
    hand, ace_in_use = process_card(hand, ace_in_use, get_card())

    if hand > 21:
        return -1, (-1, "burst", -1)
    else:
        return 0, (hand, dealer, ace_in_use)


def stick(s):
    hand, dealer_hand, ace_in_use = s
    dealer_hand = simulate_dealer(dealer_hand)

    if dealer_hand > 21 or dealer_hand < hand:
        r = 1
    elif dealer_hand > hand:
        r =  -1
    else:
        r = 0

    return r, (-1, "end", -1)

def take_determined_action(s, a):
    if a == HIT:
        r, ns = hit(s)
    else:
        r, ns = stick(s)
    return a, r, ns

def take_action(s, pi):
    a = np.random.choice((0, 1), p=pi[s])
    return take_determined_action(s, a)

def generate_episode(s, pi):
    episode=[]
    while True:
        a, r, ns = take_action(s, pi)
        episode.append((s, a, r))
        if ns[1] in ('burst', 'end'):
            return episode
        s = ns

def get_start_state():
    hand = 0; usable_ace = False
    hand, usable_ace = process_card(hand, usable_ace, get_card())
    hand, usable_ace = process_card(hand, usable_ace, get_card())
    dealer = get_card()
    if hand == 21:
        start = (hand, dealer, usable_ace)
        if dealer + get_card() == 21:
            return (start, "tie", 0)
        else:
            return (start, "natural", 1)

    while hand <= 11:
        hand, usable_ace = process_card(hand, usable_ace, get_card())

    return (hand, dealer, usable_ace)

# return game episode
def play_game(pi):
    start = get_start_state()
    if start[1] in ("tie", "natural"):
        return [start]
    return generate_episode(start, pi)