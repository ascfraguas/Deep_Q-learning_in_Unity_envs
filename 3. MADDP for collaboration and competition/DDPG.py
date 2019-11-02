
import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
# ------------------- ELEMENTS OF OUR AGENT --------------------------#
# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #



########################
#### REPLAY BUFFER #####
########################

class ReplayBuffer:

    ''' Fixed-size buffer to store experience tuples '''

    def __init__(self, action_size, buffer_size=150, batch_size=20, seed=42):

        ''' Initialize buffer parameters '''

        # Initialization parameters
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

        # Empty object to store experiences
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])


    def add(self, state, action, reward, next_state, done):

        ''' Add a new experience to memory '''
        self.memory.append(self.experience(state, action, reward, next_state, done))


    def sample(self):

        ''' Randomly sample a batch of experiences from memory '''

        # Sample k experiences out of the memory
        experiences = random.sample(self.memory, k=self.batch_size)

        # Get the replay memory for the two agents
        states = (torch.from_numpy(np.vstack([list(e.state[0])+list(e.state[1]) for e in experiences if e is not None])).float().to(device), torch.from_numpy(np.vstack([e.state[0] for e in experiences if e is not None])).float().to(device), torch.from_numpy(np.vstack([e.state[1] for e in experiences if e is not None])).float().to(device))
        
        actions = (torch.from_numpy(np.vstack([list(e.action[0])+list(e.action[1]) for e in experiences if e is not None])).float().to(device), torch.from_numpy(np.vstack([e.action[0] for e in experiences if e is not None])).float().to(device), torch.from_numpy(np.vstack([e.action[1] for e in experiences if e is not None])).float().to(device))
          
        rewards = (torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device), torch.from_numpy(np.vstack([e.reward[0] for e in experiences if e is not None])).float().to(device), torch.from_numpy(np.vstack([e.reward[1] for e in experiences if e is not None])).float().to(device))
        
        next_states = (torch.from_numpy(np.vstack([list(e.next_state[0])+list(e.next_state[1]) for e in experiences if e is not None])).float().to(device), torch.from_numpy(np.vstack([e.next_state[0] for e in experiences if e is not None])).float().to(device), torch.from_numpy(np.vstack([e.next_state[1] for e in experiences if e is not None])).float().to(device))
                                  
        dones = (torch.from_numpy(np.vstack([np.any(e.done) for e in experiences if e is not None]).astype(np.uint8)).float().to(device), torch.from_numpy(np.vstack([e.done[0] for e in experiences if e is not None]).astype(np.uint8)).float().to(device), torch.from_numpy(np.vstack([e.done[1] for e in experiences if e is not None]).astype(np.uint8)).float().to(device))
            
        return (states, actions, rewards, next_states, dones)


    def __len__(self):

        """Return the current size of internal memory."""
        return len(self.memory)


# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
# -------------------------- AGENT NETWORKS --------------------------#
# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #


def hidden_init(layer):

    #Custom function for initialization of hidden layers
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)



###########################
###### ACTOR NETWORK ######
###########################

class Actor(nn.Module):

    ''' The Actor network defines a greedy, deterministic policy. For a given input state, the network outputs the optimal
        action to be executed by the agent. Selection is driven by optimizing over Q-values provided by the critic '''

    def __init__(self, state_size, action_size, seed=None):

        ''' Initialization of the network '''

        super(Actor, self).__init__()

        # Initialize a seed for reproducibility
        if seed: self.seed = torch.manual_seed(seed)

        # Define network elements, inputs are the variables representing current state
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)

        # Reset the parameters
        self.reset_parameters()

    def reset_parameters(self):

        ''' Internal function for parameter resetting '''

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):

        ''' Forward pass to generate optimal action from input state '''
        
        x = F.elu(self.fc1(state))
        x = F.elu(self.fc2(x))
        return torch.tanh(self.fc3(x))



############################
###### CRITIC NETWORK ######
############################

class Critic(nn.Module):

    ''' The Critic network estimates Q-values for (s,a) pairs. The training experiences are those collected by the Actor network '''

    def __init__(self, state_size, action_size, seed=None):

        ''' Initialization of the network '''

        super(Critic, self).__init__()

        # Initialize a seed for reproducibility
        if seed: self.seed = torch.manual_seed(seed)

        # Define network elements, inputs are the variables representing current state along with the action tuple
        self.fc1 = nn.Linear(state_size*2, 256)
        self.fc2 = nn.Linear(256+action_size, 256)
        self.fc3 = nn.Linear(256, 1)

        # Reset the parameters
        self.reset_parameters()

    def reset_parameters(self):

        ''' Internal function for parameter resetting '''

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):

        ''' Forward pass to estimate Q-value of the input (s,a) pair '''

        x = F.elu(self.fc1(state))
        x = torch.cat((x, action), dim=1)
        x = F.elu(self.fc2(x))
        return self.fc3(x)




# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
# -------------------------- LEARNING AGENT --------------------------#
# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #




###################################
###### AGENT HYPERARAMETERS  ######
###################################

BUFFER_SIZE = int(1e4)                   # replay buffer size
BATCH_SIZE = 128                         # minibatch size
GAMMA = 0.99                             # discount factor
TAU = 1e-3                               # for soft update of target parameters
LR_ACTOR, LR_CRITIC = 2e-4, 3e-4         # learning rate of the networks
UPDATE_EVERY = 1                         # how often to update the network


# Indicate where to make the computations, either CPU or GPU (if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



########################
###### THE AGENT  ######
########################


class Agent():

    ''' The DDPG agent uses the above defined networks to optimize its interaction with the environment '''

    def __init__(self, agent_index, state_size, action_size, seed=None):

        ''' Initialization of the agent '''

        # Initialize state / action space sizes, and the counter for updating
        self.agent_index = agent_index
        self.state_size = state_size
        self.action_size = action_size
        if seed: self.seed = random.seed(seed)
        self.t_step = 0

        # Initialize the Actor networks (i.e. local and target)
        self.actor_local = Actor(state_size, action_size, self.seed).to(device)
        self.actor_target = Actor(state_size, action_size, self.seed).to(device)
        for targ, loc in zip(self.actor_target.parameters(), self.actor_local.parameters()): targ.data.copy_(loc.data)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Initialize the Critic networks (i.e. local and target)
        self.critic_local = Critic(state_size, action_size, self.seed).to(device)
        self.critic_target = Critic(state_size, action_size, self.seed).to(device)
        for targ, loc in zip(self.critic_target.parameters(), self.critic_local.parameters()): targ.data.copy_(loc.data)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Initialize agent's replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, self.seed)


    def step(self, agent_index, state, action, reward, next_state, done):

        ''' Store experience and learn if it is time to do so '''

        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:

            if len(self.memory) > BATCH_SIZE: self.learn(GAMMA)


    def act(self, state, eps=0.):

        ''' Returns actions for given state as per current policy (Actor network) '''

        state = torch.from_numpy(state).unsqueeze(0).float().to(device)

        # Compute the optimal action
        self.actor_local.eval()
        with torch.no_grad(): action = self.actor_local(state).squeeze()
        self.actor_local.train()

        # We add random noise to simulate epsilon-greedy selection. Noise centered around 0, mean of the distribution shifted according
        # the selected value. Use snippet below to understand noise distribution
        '''
        import matplotlib.pyplot as plt
        v=-0.9
        epsilon=0.01
        plt.hist(epsilon*(np.random.weibull(1+v/2,1000)-np.random.weibull(1-v/2,1000)), bins=100)
        '''
        noise = torch.Tensor([eps*(np.random.weibull(1+np.float(x)/2)-np.random.weibull(1-np.float(x)/2)) for x in action])
        return np.clip(action+noise, -1, 1)


    def soft_update(self, local_model, target_model, tau):

        ''' Soft update for the target networks '''

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



    def learn(self, gamma):

        ''' Update local network weights using sampled experience tuples (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples '''

        
        # --- SAMPLE EXPERIENCES
        
        # Sample
        experiences = self.memory.sample()
        
        # Experiences for actor
        states      = experiences[0][self.agent_index]
        actions     = experiences[1][self.agent_index]
        rewards     = experiences[2][self.agent_index]
        next_states = experiences[3][self.agent_index]
        dones       = experiences[4][self.agent_index]
        
        # Experiences for critic
        states_for_critic      = experiences[0][0]
        actions_for_critic     = experiences[1][0]
        rewards_for_critic     = experiences[2][0]
        next_states_for_critic = experiences[3][0]
        dones_for_critic       = experiences[4][0]
        
        
        # --- CRITIC UPDATE
           
        # Get actions in next_states (Actor)
        actions_next = self.actor_target(next_states)

        # Estimate boostrapped Q-values of (s, a) pairs using estimated Q-values for (next_s, next_a) pairs and actual rewards
        Q_targets_next = self.critic_target(next_states_for_critic, actions_next)
        #Q_targets = rewards_for_critic.sum(dim=1).unsqueeze(dim=1) + (gamma * Q_targets_next * (1 - dones_for_critic))
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones_for_critic))
        
        # Get the estimated Q-values for the states and calculate loss function
        Q_expected = self.critic_local(states_for_critic, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Gradient descent step
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
        self.critic_optimizer.step()

        
        # --- ACTOR UPDATE
        
        # Compute actor loss as minus the mean Q-value across (s, a) pairs
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states_for_critic, actions_pred).mean()

        # Gradient descent step
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- TARGET NETWORKS

        # Update target network
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU)
