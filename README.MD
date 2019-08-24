# Deep Q-Learning on Unity environments

Unity [ML-agents](https://github.com/Unity-Technologies/ml-agents) provides a powerful framework for training
and testing AI agents on different environments.

-----------

## 1. The project

In this section the environment and the agent's are introduced, along with the results of the training process.


### 1.1. The environment

In this project we train an agent to optimize its interaction with a prebuilt environment. The agent, which has 
4 possible available actions (forward, backward, left and right), has to collect as many yellow bananas as possible
avoiding blue bananas in each run of the environment (i.e. episode).

### 1.2. The agent

The agent in this project is trained using a vanilla version of the deep Q-network algorithm (refer to this 
[research paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf)). 

For more detail on the agent's configuration, refer to the file `dqn_agent.py`, described in detail in section 2.2 of this document.

### 1.3. Results 

The file `DQN in the Banana environment.ipynb` hosts detailed description of the agent and its interaction
with the environment, as well as the training algorithm and the results.

The agent solves the environment (i.e. score over 13 in the past 100 episodes) in a total of XXXXX episodes.

While the current agent's specification is considered to adequately solve the environment, several improvements have 
been identified as applicable to the algorithm, including double Q-networks, prioritized replay and dueling Q-networks.


-----------

## 2. Working with this project

In this section I describe the technical requirements of this project, along with instructions on how to peruse available files.


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


### 2.2. Perusing the contents of this project

The file `DQN in the Banana environment.ipynb` walks the user through the entire project. We can divide it in the following sections:

1. **The environment and the agent**: Sections 1 and 2 introduce these concepts and enable the user to see a example episode, with actions chosen at random
2. **Training the agent**: Section 3 runs the algorithm for training the agent, displaying the results of the training process. This step creates the file `checkpoint.pth`, which contains the weights of the agent's optimized Q-network
3. **The agent in action**: Section 4 leverages the results from the training process stored in the file `checkpoint.pth` to initialize a trained agent, which we can see in action!

Although the entire project is run from `DQN in the Banana environment.ipynb`, we can also fin the following files in the project:

- `Banana_Windows_x86_64`: This file contains the pre-built Unity environment, operative for Windows 10 64B.
- `dqn_agent.py`: This file details the classes we leverage on to build and train our agent. These are:
	- `ReplayBuffer`: This class stores the uncorrelated agent's experiences, randomly sampled during the updating of the agent's network
	- `Q-Network`: The agent's Q-network architecture is composed of 3 hidden, fully connected layers of 256, 512, and 256 nodes each with ReLU activationfunctions. Input layer takes in 37 sensory inputs delivered by the agent, and the output layer includes 4 nodes to model the Q-values of each one of the 4 actions available to the agent.
	- `Agent`: The agent's class. This class features a `ReplayBuffer` and two `Q-Network` attributes, as well as a variety of methods used to pick actions and learn from previous actions.