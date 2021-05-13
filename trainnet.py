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

from mcts import MCTS
from nnet import *
from connect4 import *
from trainer import train

# This function generates data, uses that data to train new model, and saves the new model to
# the Models folder for the total number of rounds desired. You must have a "Models" folder in
# the current working directory!

class train_model():
    def __init__(self, mcts, nnet):
        self.data = []
        self.mcts = mcts
        self.nnet = nnet
        self.recent_game = None

    def policyIteration(self, start_round, rounds, episodes, iterations, dup):
        for i in range(start_round, start_round + rounds + 1):

            net = self.nnet
            self.mcts = MCTS(net, iterations)
            mcts = self.mcts
            print("ROUND")
            print(i)
            path = "Models/checkpoint" + "_" + str(i) + "_" + str(episodes) + "_" + str(mcts.iterations) + "_" + str(dup) + ".pth"
            print("model " + path + " saved")
            torch.save(net.state_dict(), path)
            state_dict = torch.load(path)
            net.load_state_dict(state_dict)
            
            if i >= rounds:
                return self.nnet
            
            for e in range(episodes):
                print(e)
                self.data += self.executeEpisode()       # collect examples from this game
                print(len(self.data))
            
            if dup:
                duplicate =  [(encode_reverse(x[0]), x[1], x[2]) for x in self.data]
                self.data += duplicate
            
            datasets = np.array(self.data)
            optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.8, 0.999))
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150,200,250,300,400], gamma=0.77)
            train(net, datasets, optimizer, scheduler, 0, 0, 0)
            self.nnet = net
            self.data = []
        
        return self.nnet


    def executeEpisode(self):
        mcts = self.mcts
        nnet = self.nnet
        examples = []
        s = Connect4()
        i = 0
        while True:
            i+= 1
            self.recent_game = s
            mcts.Qsa, mcts.Nsa, mcts.Ns, mcts.Es, mcts.Vs = {}, {}, {}, {}, {}
            
            if i < 20:
                probs = mcts.getActionProb(game = s)
            else:
                probs = mcts.getActionProb(game = s, thresh=0)
            
            encoded = encode(s)
            examples.append([encoded, probs, s.turn, None])                   # rewards can not be determined yet 
            action = np.random.choice(len(probs), p=probs)
            s = s.result(action)
            print(s.print_board())
            if s.game_over():
                return [(x[0], np.asarray(x[1]), float(s.win_move() * ((-1) ** (x[2] != examples[-1][2])))) for x in examples]