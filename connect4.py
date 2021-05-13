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

# This is an implementation of a game of Connect Four with a board as a 2D np array

# CONSTANTS
ROW_COUNT = 6
COLUMN_COUNT = 7
CONNECT_TOTAL = 4
C_PUCT = 4
P1BOARD = np.ones((ROW_COUNT,COLUMN_COUNT))
P2BOARD = np.ones((ROW_COUNT,COLUMN_COUNT))*2
ITERATIONS = 25
EPS = 1e-8
NUMEPISODES = 50
NUMROUNDS = 1


def next_player(curr):
    if curr == 1: return 2
    else: return 1

class Connect4:
    def __init__(self):
        self.board = np.zeros((ROW_COUNT,COLUMN_COUNT))
        self.turn = 1
        self.w = 0
        self.most_recent = [0,0]
    
    # check if there is room to drop a piece
    def is_valid_location(self, col):
        if col < 0 or col >= COLUMN_COUNT:
            return 0
        return self.board[0][col] == 0
    
    # return legal moves
    def legal_moves(self):
        #return np.where((self.board[ROW_COUNT-1] == 0) == True)[0]
        valid_locations = []
        for col in range (COLUMN_COUNT):
            if self.is_valid_location(col):
                valid_locations.append(col)
        return valid_locations

    def valid_moves(self):
        # valid = (self.board[ROW_COUNT-1,] == 0)
        # return valid.astype(int) #*list(range(7))
        valid_locations = []
        for col in range (COLUMN_COUNT):
            if self.is_valid_location(col):
                valid_locations.append(1)
            else:
                valid_locations.append(0)
        return np.array(valid_locations)
    
    # get row to put next piece for a certain column
    def get_next_open_row(self, col):
        for r in range(ROW_COUNT-1, -1, -1):
            if self.board[r][col] == 0:
                return r
            
    def print_board(self):
        #return np.flip(self.board, 0)
        return(self.board)
        
    def result(self, col):
        if (self.is_valid_location(col) == 0): assert(1==2)
        row = self.get_next_open_row(col)
        piece = self.turn
        self.board[row][col] = piece
        self.most_recent = [row,col]
        self.winning_move(piece) #print("player", piece, "wins!")
        self.turn = next_player(piece)
        return self
    
    def winning_move(self, piece):
        r = self.most_recent[0]
        c = self.most_recent[1]

        left = max(c - CONNECT_TOTAL, 0)
        right = min(c + CONNECT_TOTAL, COLUMN_COUNT)
        top = max(r - CONNECT_TOTAL, 0)
        bottom = min(r + CONNECT_TOTAL, ROW_COUNT)

        #check left to right
        consecutive = 0
        for i in range(left,right):
            if self.board[r][i] == piece:
                consecutive +=1
            else: 
                consecutive = 0
            if consecutive == CONNECT_TOTAL:
                if self.w == 0: self.w = piece
                return True
        
        consecutive = 0
        #check top to bottom
        for i in range(top,bottom):
            if self.board[i][c] == piece:
                consecutive +=1
            else: 
                consecutive = 0
            if consecutive == CONNECT_TOTAL:
                if self.w == 0: self.w = piece
                return True
        
        #check positive-slope diagnonal
        consecutive = 0
        for i in range(-4,5):
            if (c+i >= 0) and (r-i < ROW_COUNT):
                if (c+i < COLUMN_COUNT) and (r-i >= 0):
                    if self.board[r-i][c+i] == piece:
                        consecutive +=1
                    else:
                        consecutive = 0
                    if consecutive == CONNECT_TOTAL:
                        if self.w == 0: self.w = piece
                        return True
        
        r = self.most_recent[0]
        c = self.most_recent[1]

        #check negative-slope diagnonal
        consecutive = 0
        for i in range(-4,5):
            if (c+i >= 0) and (r+i < ROW_COUNT):
                if (c+i < COLUMN_COUNT) and (r+i >= 0):
                    if self.board[r+i][c+i] == piece:
                        consecutive +=1
                    else:
                        consecutive = 0
                    if consecutive == CONNECT_TOTAL:
                        if self.w == 0: self.w = piece
                        return True
      
    def game_over(self):
        if self.winning_move(1):
            return True
        if self.winning_move(2):
            return True
        if len(self.legal_moves()) == 0:
            self.w = 0.0000001
            return True

    def win_move(self):
        if self.winning_move(1) == 1 or self.winning_move(2) == 1:
            return 1
        return 0
    
    def winner(self):
        return self.w
    
    def next_player(self):
      if self.turn == 1: return 2
      else: return 1
    
    def stringRepresentation(self):
        return (str(self.board) + str(self.turn))

# Encodes the board into a 3D tensor
def encode(game):
    turn = game.turn
    board = game.print_board()
    board_1 = np.where(board == 2, 0, board)
    board_2 = np.where(board == 1, 0, board)
    player_board = P1BOARD if turn == 1 else P2BOARD
    encoded_board = [board_1, board_2, player_board]
    return torch.tensor(encoded_board)

# Flips an encoded board
def encode_reverse(board):
    copy = np.ndarray.copy(np.flip(np.asarray(board), (2)))
    return torch.from_numpy(copy)

# Random choice strategy
def random_choice(position):
    moves = position.legal_moves()
    return random.choice(moves)