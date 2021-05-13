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
import matplotlib
import matplotlib.pyplot as plt
import os
import re

from connect4 import Connect4
from mcts import *
from trainer import *
from trainnet import *
from nnet import *
from pit import *

# This function is the primary driver of Alpha Connect Four
# 

def find_most_recent_model(episodes, iterations, dup):
    files = os.listdir("/home/accts/hrc24/seniorproject/Models")
    if files.count(".DS_Store"):
        files.remove(".DS_Store")

    if files.count("BASEMODEL.pth"):
        files.remove("BASEMODEL.pth")
    
    if len(files) == 0:
        return [0,"Models/BASEMODEL.pth"]

    files = [re.sub("checkpoint_", "", x) for x in files]
    files = [re.sub(".pth", "", x) for x in files]
    files = [x.split("_") for x in files]
    files = [[int(x) for x in y] for y in files]
    files = np.asarray(files)
    
    files = files[(files[:,1] == episodes),:]
    files = files[(files[:,2] == iterations),:]
    files = files[(files[:,3] == dup),:]

    if (len(files) == 0):
        return [0,"Models/BASEMODEL.pth"]
    else:
        file = files[np.argmax(files[:,0]),:]
        return [file[0],"Models/checkpoint_" + str(file[0]) + "_" + str(file[1]) + "_" + str(file[2]) + "_" + str(file[3]) + ".pth"]


def main():
    net = ConnectNet()
    net = net.double()
    torch.save(net.state_dict(), "/home/accts/hrc24/seniorproject/Models/BASEMODEL.pth")

    rounds = 5
    episodes = 200
    iterations = 400
    dup = 1
    most_recent = find_most_recent_model(episodes, iterations, dup)
    state_dict = torch.load(most_recent[1])
    net.load_state_dict(state_dict)
    current_model = 0
    mcts = MCTS(net, iterations)
    model = train_model(mcts, net)
    mod = model.policyIteration(current_model, rounds, episodes, iterations, dup)

if __name__ == "__main__":
    if len(sys.argv) == 8:
        rounds = int(sys.argv[1])           # Total rounds of training to perform
        episodes = int(sys.argv[2])         # Total number of episodes to be run
        iterations = int(sys.argv[3])       # Total number of iterations
        dup = int(sys.argv[4])              # Duplicate data using symmetry
        method = sys.argv[5]                # Either train, pit, or sample test
        strat1 = sys.argv[6]                # Either nn or random
        strat2 = sys.argv[7]                # Either nn or random

        # Train a new model
        if method == "train":
            net = ConnectNet()
            net = net.double()
            most_recent = find_most_recent_model(episodes, iterations, dup)
            state_dict = torch.load(most_recent[1])
            net.load_state_dict(state_dict)
            current_model = most_recent[0]
            mcts = MCTS(net, iterations)
            model = train_model(mcts, net)
            mod = model.policyIteration(current_model, rounds, episodes, iterations, dup)
        
        # Pit models against one another
        if method == "pit":
            net = ConnectNet()
            net = net.double()
            if strat1 == "random" and strat2 == "random":
                compare = arena(random_choice, random_choice, random, random)
                compare.pit(episodes)

            if strat1 == "random" and strat2 != "random":
                state_dict = torch.load(strat2)
                net.load_state_dict(state_dict)
                mcts = MCTS(net, iterations)
                compare = arena(random_choice, mcts, random, "nn")
                compare.pit(episodes)
            
            if strat1 != "random" and strat2 == "random":
                state_dict = torch.load(strat1)
                net.load_state_dict(state_dict)
                mcts = MCTS(net, iterations)
                compare = arena(mcts, random_choice, "nn", random)
                compare.pit(episodes)
            
            else:
                state_dict = torch.load(strat1)
                net.load_state_dict(state_dict)
                net1 = copy.deepcopy(net)
                mcts1 = MCTS(net1, iterations)
                state_dict = torch.load(strat2)
                net.load_state_dict(state_dict)
                mcts2 = MCTS(net, iterations)
                compare = arena(mcts1, mcts2, "nn", "nn")
                compare.pit(episodes)
            
            print(compare.s1wins)
            print(compare.s2wins)
        
        # For testing the output of different neural networks
        else:
            state_dict = torch.load('Models/checkpoint_3_250_800_0.pth')
            net.load_state_dict(state_dict)
            game = Connect4()
            print(net(encode(game)))
