
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
        
        # Process the different memory elements and return them
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    
    def __len__(self):
        
        """Return the current size of internal memory."""
        return len(self.memory)
    

    
    
########################
###### Q-NETWORK  ######
########################


class QNetwork(nn.Module):
    
    ''' Actor (Policy) Model '''

    def __init__(self, state_size, action_size, seed):
        
        ''' Initialization of parameters '''
        
        # Main parameters
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)
        
        # Define elements network
        self.criterion = nn.MSELoss() 
        self.hidden_1 = nn.Linear(self.state_size, 256)
        self.hidden_2 = nn.Linear(256, 512)
        self.hidden_3 = nn.Linear(512, 256)
        self.output = nn.Linear(256, self.action_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        #self.dropout = nn.Dropout(p=0.1)
        #self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, state):
        
        # Pass the input tensor through each of our operations
        x = self.relu(self.hidden_1(state))
        x = self.relu(self.hidden_2(x))
        x = self.relu(self.hidden_3(x))
        x = self.output(x)
        
        return x

    
    
# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
# -------------------------- LEARNING AGENT --------------------------#
# ------------------------------------------------------------------- #
# ------------------------------------------------------------------- #
   



###################################
###### AGENT HYPERARAMETERS  ######
###################################


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

# Indicate where to make the computations, either CPU or GPU (if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



########################
###### THE AGENT  ######
########################


class Agent():

    
    def __init__(self, state_size, action_size, seed):
        
        ''' Initialization of the agent '''
        
        # Initialize state / action space sizes, and the counter for updating
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.t_step = 0

        # Initialize the two Q-networks (the local and target) and define the optimizer
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Initialize agent's replay buffer
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
    
    def step(self, state, action, reward, next_state, done):
        
        ''' Store experience and learn if it is time to do so '''
        
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

                
                
    def act(self, state, eps=0.):
        
        ''' Returns actions for given state as per current policy '''
            
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Set the Q-network to evaluation mode (turn off training layers such as dropouts) and do a forward pass for the state
        # without computing gradients. Afterwards, return to training mode
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps: return np.argmax(action_values.cpu().data.numpy())
        else: return random.choice(np.arange(self.action_size))

        
        
    def soft_update(self, local_model, target_model, tau):
        
        ''' Soft update for the target Q-network '''
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

            
            
    def learn(self, experiences, gamma):
        
        ''' Update local network weights using sampled experience tuples (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples '''
        
        # Unpack experiences
        states, actions, rewards, next_states, dones = experiences
        
        # Get predictions from local network, i.e. Q-values of the (state, action) pairs in the sampled batch of experiences
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        # Get training targets, i.e. current reward + max predicted Q-value for next state by target network, for each (state, action)
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Now compute the loss wrt these new targets
        loss = self.qnetwork_local.criterion(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     



    