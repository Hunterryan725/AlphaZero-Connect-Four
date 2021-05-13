import numpy as np
import sys
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import math
import pdb
import pandas as pd
from torch.nn.utils import clip_grad_norm_
import matplotlib.pyplot as plt
import os
import re

from connect4 import *

# This is the function that runs the Monte Carlo Tree Search in the Alpha Zero protocol. 
# This implementation is based heavily on https://web.stanford.edu/~surag/posts/alphazero.html
# implementation with various modifications for performance boost and correction

class MCTS():
    def __init__(self, nnet, iterations):
        self.nnet = nnet
        self.iterations = iterations
        self.Qsa = {}   # stores Q values for s,a 
        self.Nsa = {}   # stores the number of times edge s,a was visited
        self.Ns = {}    # stores the number of times board s was visited
        self.Ps = {}    # stores initial policy (returned by neural net)
        self.val = {}   # store  initial value (returned by neural net)
        self.Es = {}    # stores game.getGameEnded ended for board s
        self.Vs = {}    # stores game.getValidMoves for board s
        self.iters = 0
        
    # Runs searches equivalent to the number of iterations and returns the 
    # tree searches action probability to train the next network on
    def getActionProb(self, game, thresh=1):

        for i in range(self.iterations):
            game2 = copy.deepcopy(game)
            self.search(game2)

        s = game.stringRepresentation()
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(COLUMN_COUNT)]

        if thresh == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        else:
            counts = [x ** (1. / thresh) for x in counts]
            counts_sum = float(sum(counts))
            probs = [x / counts_sum for x in counts]
            return probs

    # Runs a single search
    def search(self, game):
        s = game.stringRepresentation()

        if s not in self.Es:
            v = game.game_over() 
            self.Es[s] = game.w
        
        if self.Es[s] != 0: # We know this is a terminal node
            return -self.Es[s]

        if s not in self.Vs:
            valids = game.valid_moves()
            if s not in self.Ps: # If we have not run nn on the board position add it to the hashtable
                encoded_board = encode(game)
                policy, value = self.nnet(encoded_board)
                v = value.item()
                self.Ps[s] = policy.tolist()[0] * valids

                summed = np.sum(self.Ps[s])
                if summed > 0:
                    self.Ps[s] /= summed
                else:
                    self.Ps[s] = self.Ps[s] + valids
                    if(np.sum(self.Ps[s]) == 0): assert(5 == 2)
                    self.Ps[s] /= np.sum(self.Ps[s])
                self.val[s] = v
            
            v = self.val[s]
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v
        
        if s not in self.Vs:
            assert(1 == 2)
        
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        for a in range(COLUMN_COUNT):
            if valids[a]:
                if (s,a) in self.Qsa:
                    u = self.Qsa[(s, a)] + C_PUCT * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = C_PUCT * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
                if u > cur_best:
                    cur_best = u
                    best_act = a

        self.Ns[s] += 1
        a = best_act
        next_board = game.result(a)

        v = self.search(next_board)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        
        return -v
