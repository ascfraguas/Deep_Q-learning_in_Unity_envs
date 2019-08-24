# Deep Q-Learning on Unity environments

-----------

## 1. Overview

Unity [ML-agents](https://github.com/Unity-Technologies/ml-agents) provides a powerful framework for training
and testing AI agents on different environments. 

In this project we train an agent to optimize its interaction with a prebuilt environment, a 3D world in which blue and
yellow bananas spawn at random. The agent, which has 4 possible available actions (moving forward, backward, left and right), 
has to collect as many yellow bananas as possible avoiding blue bananas in each run of the environment (i.e. episode).

See below an actual run of the environment by the succesfuly trained agent:

<img src="images/agent_in_action.gif" width="400" height="280" loc="center"/>


The agent in this project is trained using a vanilla version of the deep Q-network algorithm (refer to this 
[research paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)). 
Section 4 for technical detail on the agent's configuration and the training algorithm.


-----------

## 2. Working with this project

In this section I describe the technical requirements of this project, along with instructions on how to use available files.


### 2.1. Environment setup

To set up your environment to run this project, follow the below commands from your Anaconda prompt - assuming
you are running on a Windows 10 machine with Anaconda already installed.

**NOTE**: Version conflicts found when pip installing from available wheels created issues when pip installing
the environment from a requirements.txt file. Below commands solve this issue.

1. `conda create -n banana_env python==3.6`
2. `activate banana_env` 
3. `conda install pytorch=1.0.1`
4. `pip install unityagents==0.4.0`

After this you can launch your Jupyter kernel with `jupyter notebook`, and ready to go!


### 2.2. Using the contents of this project

The file `DQN in the Banana environment.ipynb` walks the user through the entire project. We can divide it in the following sections:

1. **The environment and the agent**: Sections 1 and 2 introduce these concepts and enable the user to see a example episode, with actions chosen at random
2. **Training the agent**: Section 3 runs the algorithm for training the agent, displaying the results of the training process. This step creates the file `checkpoint.pth`, which contains the weights of the agent's optimized Q-network
3. **The agent in action**: Section 4 leverages the results from the training process stored in the file `checkpoint.pth` to initialize a trained agent, which we can see in action!

Although the entire project is run from `DQN in the Banana environment.ipynb`, we can also find the following files in the project:

- `Banana_Windows_x86_64`: This folder contains the pre-built Unity environment, operative for Windows 10 64B.
- `dqn_agent.py`: This file details the classes we leverage on to build and train our agent. These are:
	- `ReplayBuffer`: This class stores the uncorrelated agent's experiences, randomly sampled during the updates of the agent's network
	- `Q-Network`: The agent's Q-network architecture.
	- `Agent`: The agent's class. This class features a `ReplayBuffer` and two `Q-Network` attributes, as well as a variety of methods used to pick actions and learn from previous experience.


See the next section for technical detail on the implemented algorithm.


-----------

## 3. Technical report

This section is the **technical report** for the project, detailing implementation and results


### 3.1. The algorithm

The agent is initialized with two neural networks (i.e. local and target) with the same architecture. The architecture of the 
network is composed of 3 hidden, fully connected layers of 256, 512, and 256 nodes each with ReLU activation functions. Input 
layer takes in 37 sensory inputs delivered by the agent, and the output layer includes 4 linear nodes to model the Q-values of 
each one of the 4 actions available to the agent.

The agent is trained for a total of XXXX episodes before solving the environment, with a mean score of XXXX over the last 100 
episodes. For each time step within each episode, the agent follows the next steps:

1. **Choice of action**: The agent chooses an epsilon-greedy action w.r.t. its current policy. Specifically, the environment state
is passed onto the local network, delivering estimates for the Q-value of each possible action. The agent picks the action with
the highest Q-value with a probability `1-epsilon` where `epsilon` is the exploration parameter for the agent. This hyper-parameter
is initialized with a value of `1`, decaying a 5% of its current value in each episode up to a minimum of `0.01`.
2. 
The agent randomly samples 20 eperiences out of the buffer of 150 (state, action, reward) tuples, used to update the weights of 
the agent's local network.

The file `DQN in the Banana environment.ipynb` hosts detailed description of the agent and its interaction
with the environment, as well as the training algorithm and the results.

The agent solves the environment (i.e. score over 13 in the past 100 episodes) in a total of XXXXX episodes.

While the current agent's specification is considered to adequately solve the environment, several improvements have 
been identified as applicable to the algorithm, including double Q-networks, prioritized replay and dueling Q-networks.



 
