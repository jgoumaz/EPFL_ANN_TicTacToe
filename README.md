# EPFL - Tic Tac Toe Reinforcement Learning
**Jeremy Goumaz, Laurent Gürtler**
<br>_EPFL course | CS-456 - Artificial neural networks_
## Objectives
The goal of this project is to teach an agent to play Tic Tac Toe by Reinforcement Learning. Different methods are tested:
- Q-Learning while playing against an expert agent (`OptimalPlayer`)
- Q-Learning while playing against itself (self-practice)
- Deep Q-Learning while playing against an expert agent (`OptimalPlayer`)
- Deep Q-Learning while playing against itself (self-practice)
## Implementation
The environment used to play Tic Tac Toe is defined in the class `TictactoeEnv` and an optimal player (playing perfect moves) is defined in the class `OptimalPlayer`. `OptimalPlayer` also takes an exploration level parameter $\epsilon$ which corresponds to the proportion of random moves played ($\epsilon$-greedy policy). It means that Opt(0) will play perfectly whereas Opt(1) will play completely randomly.

We defined the Player agents in the classes `QLearningPlayer` and `DeepQLearningPlayer`:
- `QLearningPlayer` implements the basic Q-Learning algorithm with an $\epsilon$-greedy policy or alternatively with a decreasing exploration defined as $\epsilon(n) = \max\left(\epsilon_{min}, \epsilon_{max}\left(1-\frac{n}{n^*}\right)\right)$.
- `DeepQLearningPlayer` implements a Deep Q-Learning Network defined in the class `DQN`. The network is a fully-connected neural network with two hidden layers containing 128 nodes with each layer followed by a ReLU activation function. The network uses Huber Loss and is updated by Adam optimizer.

We trained the Player against `OptimalPlayer` or against itself (self-practice). We give him a reward of 1 when he wins, 0 when the game is a tie and -1 when he loses (or try to make an illegal move for `DeepQLearningPlayer` only). 

We used different metrics (evaluated each 250 games) to measure the quality of a model:
- Average reward : average reward of the last 250 games
- $M_{opt}$ : average reward against a perfect player Opt(0)
- $M_{rand}$ : average reward against a random player Opt(1)
- Average loss : average Huber loss of the last 250 games (only for `DeepQLearningPlayer`)
## Repository content
Main files:
- [tic_env.py](tic_env.py): contains the classes `TictactoeEnv` and `OptimalPlayer`
- [Players.py](Players.py): contains the classes `QLearningPlayer`,  `DeepQLearningPlayer`, `DQN` and `BufferMemory`
- [runs.py](runs.py): contains functions to train the networks and to evaluate the metrics
- [utils.py](utils.py): contains utility functions (including type conversion functions for the indices or the grids)
- [questions.ipynb](questions.ipynb): reproducible notebook containing the analysis and the figures
- [Tic_Tac_Toe.pdf](Tic_Tac_Toe.pdf): final report

Secondary files:
- [MP_TicTocToe.pdf](MP_TicTocToe.pdf): details and instructions for the project
- [tic_tac_toe.ipynb](tic_tac_toe.ipynb): notebook containing use examples of [tic_env.py](tic_env.py)
- [Figures](Figures): folder containing the plots and figures saved from [questions.ipynb](questions.ipynb)
