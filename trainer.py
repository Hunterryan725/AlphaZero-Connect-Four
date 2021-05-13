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

from nnet import *


# Trains a new neural network based on the data produced by tree search
# This is heavily based on the implementation in 
# https://towardsdatascience.com/from-scratch-implementation-of-alphazero-for-connect4-f73d4554002a
# with slight modifications

def train(net, dataset, optimizer, scheduler, start_epoch, cpu, iteration):
    net.train()
    criterion = AlphaLoss()
    epochs = 50
    batch_size = 64
    gradient_acc_steps = 1
    max_norm = 1.0
    train_set = board_data(dataset)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    losses_per_epoch = []

    # losses_per_epoch = load_results(iteration + 1)
    update_size = len(train_loader)//10

    for epoch in range(0, epochs):
        total_loss = 0.0
        losses_per_batch = []

        for i, data in enumerate(train_loader, 0):
            
            state, policy, value = data
            state = state.double()
            policy_pred, value_pred = net(state)
            loss = criterion(value_pred[:,0], value, policy_pred, policy)
            loss = loss/gradient_acc_steps
            loss.backward()
            clip_grad_norm_(net.parameters(), max_norm)

            if (epoch % gradient_acc_steps) == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()

            if i % update_size == (update_size - 1):    # print every update_size-d mini-batches of size = batch_size
                losses_per_batch.append(gradient_acc_steps*total_loss/update_size)
                print("Value (actual, predicted):", value[0].item(), value_pred[0,0].item())
                print(" ")
                total_loss = 0.0
            
        scheduler.step()
        if len(losses_per_batch) >= 1:
            losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))

    fig = plt.figure()
    ax = fig.add_subplot(222)
    ax.scatter([e for e in range(start_epoch, (len(losses_per_epoch) + start_epoch))], losses_per_epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss per batch")
    ax.set_title("Loss vs Epoch")
    plt.show()