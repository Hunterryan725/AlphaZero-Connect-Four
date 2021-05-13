# AlphaZero-Connect-Four

This is an implementation of the AlphaZero protocol developed by DeepMind to play ConnectFour. 

The driver.py file is used to call the AlphaZero protocol and takes 7 additional command line arguments

argument 1: # Total rounds of training to perform

argument 2: # Total number of episodes to be run

argument 3: # Total number of iterations

argument 4: # Duplicate data using symmetry

argument 5: # Either train or pit

argument 6: # Either nn or random

argument 7: # Either nn or random

The train protocol takes the number of rounds, episodes, and iterations and finds the most recent model trained 
and then continues to train that model. If no such model exists, a base model is loaded and training is started from
the round 1. Otherwise, training is started from the most recent round of the model that is loaded and training is 
continued for the total number of rounds. Additionally, to double the size of the train data by using symmetry, argument 4
should be set equal to 1.

To initially train a model for 5 rounds with 200 episodes and 800 iterations of MCTS with no duplication the following can be called

python3 driver.py 5 200 800 0 train random random

The pit protocol takes either:
1: a single neural network's path to be loaded and plays it against a random agent
2: two neural network paths to be loaded and played against one another

The strategies are then pitted against one another equal to the number of episodes input in the command line argument

python3 driver.py 5 250 400 1 pit Models/checkpoint_3_250_800_0.pth Models/checkpoint_3_250_800_0.pth

Special thanks to implementations of AlphaZero by https://web.stanford.edu/~surag/posts/alphazero.html and 
https://towardsdatascience.com/from-scratch-implementation-of-alphazero-for-connect4-f73d4554002a that helped tremendously
in this process
