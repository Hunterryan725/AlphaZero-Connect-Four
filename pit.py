from mcts import *
from connect4 import *

# Perform the MCTS of AlphaZero
def perform_mcts(s, mcts, j):
    if j < 20:
        probs = mcts.getActionProb(game = s)
    else:
        probs = mcts.getActionProb(game = s, thresh=0)
    return probs


# This function is used to pit various strategies against one another 
class arena():
    def __init__(self, strat1, strat2, type1, type2):
        self.strat1 = strat1
        self.strat2 = strat2
        self.type1 = type1
        self.type2 = type2
        self.s1wins = 0
        self.s2wins = 0
    
    
    def pit(self, numgames):
        
        strat1 = self.strat1
        strat2 = self.strat2
        
        for i in range(numgames):
            print(i)
            s = Connect4()
            j = 0
            while not s.game_over():
                j+= 1
                if self.type1 != "random":
                    strat1.Qsa, strat1.Nsa, strat1.Ns, strat1.Es, strat1.Vs = {}, {}, {}, {}, {}
                if self.type2 != "random":
                    strat2.Qsa, strat2.Nsa, strat2.Ns, strat2.Es, strat2.Vs = {}, {}, {}, {}, {}

                if (i % 2 == 0): # Strategy 1 first
                    if s.turn == 1: # even games mcts2 is P2
                        if self.type1 == "nn":
                            probs = perform_mcts(s, strat1, j)
                            action = np.random.choice(len(probs), p=probs)
                        else:
                            action = strat1(s)
                    else:
                        if self.type2 == "nn":
                            probs = perform_mcts(s, strat2, j)
                            action = np.random.choice(len(probs), p=probs)
                        else:
                            action = strat2(s)
                else:                 
                    if s.turn == 1: # odd games mcts2 is P1
                        if self.type2 == "nn":
                            probs = perform_mcts(s, strat2, j)
                            action = np.random.choice(len(probs), p=probs)
                        else:
                            action = strat2(s)
                    else:
                        if self.type1 == "nn":
                            probs = perform_mcts(s, strat1, j)
                            action = np.random.choice(len(probs), p=probs)
                        else:
                            action = strat1(s)


                s = s.result(action)
                print(s.board)

            if s.w == 1 and i % 2 == 0:  # if p1
                self.s1wins +=1
            elif s.w == 2 and i % 2 == 1:
                self.s1wins +=1
            else:
                self.s2wins +=1