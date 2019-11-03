import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.hidden_1 = nn.Linear(self.state_size, 64)
        self.hidden_2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, self.action_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        #self.dropout = nn.Dropout(p=0.1)
        #self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, state):
        
        # Pass the input tensor through each of our operations
        x = self.relu(self.hidden_1(state))
        x = self.relu(self.hidden_2(x))
        x = self.output(x)
        
        return x

