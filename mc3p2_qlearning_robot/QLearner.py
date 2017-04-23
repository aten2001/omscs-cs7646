"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand


class QLearner(object):
    def author(self):
        return 'jlee3259'

    def __init__(self, \
                 num_states=100, \
                 num_actions=4, \
                 alpha=0.2, \
                 gamma=0.9, \
                 rar=0.5, \
                 radr=0.99, \
                 dyna=0, \
                 verbose=False):

        self.num_states = num_states
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.table = np.random.uniform(-1.0, 1.0, size=(num_states, num_actions))

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        # (1-alpha)*Q[s,a] + alpha (r + gamma*Q[S',argmax_a'(Q[S',A'])])
        arg_max_a_prime = np.amax(self.table[s, :])
        a_prime = np.where(self.table[s, :] == arg_max_a_prime)[0]

        rand_action = rand.randint(0, self.num_actions - 1)
        rand_prob = np.random.uniform(0.0, 1.0, 1)
        action = rand_action if rand_prob <= self.rar else a_prime
        if self.verbose: print "s =", s, "a =", action
        self.a = action
        return action

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        #last taken action
        a = self.a

        #current state
        s = self.s
        # (1-alpha)*Q[s,a] + alpha (r + gamma*Q[S',argmax_a'(Q[S',A'])])
        old_val = (1 - self.alpha) * self.table[s, a]
        arg_max_a_prime = np.amax(self.table[s_prime, :])
        a_prime = np.where(self.table[s_prime] == arg_max_a_prime)[0]

        # improved estimate
        new_val = self.alpha * (r + (self.gamma * self.table[s_prime, a_prime]))
        result = old_val + new_val
        # update the Q table
        self.table[s, a] = result

        rand_action = rand.randint(0, self.num_actions - 1)
        rand_prob = np.random.uniform(0.0, 1.0, 1)
        action = rand_action if rand_prob <= self.rar else a_prime

        # decay
        self.rar = self.rar * self.radr

        if self.verbose: print "s =", s_prime, "a =", action, "r =", r

        #update the state
        self.s = s_prime

        #update the action
        self.a = action
        return action


if __name__ == "__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
