# Deep Q-Learning on Unity environments

Unity [ML-agents](https://github.com/Unity-Technologies/ml-agents) provides a powerful framework for training
and testing AI agents on different environments.

## The environment

In this project we train an agent to optimize its interaction with a prebuilt environment. The agent, which has 
4 possible available actions (forward, backward, left and right), has to collect as many yellow bananas as possible
avoiding blue bananas in each run of the environment (i.e. episode).

## The agent

The agent in this project is trained using a deep Q-network (refer to this [research paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf))
with custom enhancements, such as BLAHBLAHBLAH. The file `dqn_agent.py` hosts the code to the agent's configuration.

## Results 

The file `DQN in the Banana environment.ipynb` hosts detailed description of the agent and its interaction
with the environment, as well as the the training algorithm and the results.

The agent solves the environment (i.e. score over 13 in the past 100 episodes) in a total of XXXXX episodes.




-----------

## Environment setup

To set up your environment to run this project, follow the below commands from your Anaconda prompt - assuming
you are running on a Windows 10 machine with Anaconda already installed.

**NOTE**: Version conflicts found when pip installing from available wheels created issues when pip installing
the environment from a requirements.txt file. Below commands solve this issue.

`conda create -n banana_env python==3.6`
`activate banana_env` 
`conda install pytorch=1.0.1`
`pip install unityagents==0.4.0`

After this you can launch your Jupyter kernel with `jupyter notebook`, and ready to go!


