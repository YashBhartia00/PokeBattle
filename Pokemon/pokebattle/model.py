import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque 
import random 

class QNetwork(nn.Module):
    """ Policy Model."""
    def __init__(self, state_size,action_size, seed):
        """
        Initialize parameters and build model.
        Parameters
        -------
            state_size : Dimensions of state
            action_size : Dimensions of action
            seed : Random seed
        """
        
        super(QNetwork,self).__init__() ## calls __init__ method of nn.Module class
        self.seed = torch.manual_seed(seed)
        self.fc1= nn.Linear(state_size,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,action_size)
        
    def forward(self,x):
        """
        Builds a network that maps state to action values.
        
        Parameter
        -------
        x : state 
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    """Fixed -size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object
        
        Parameters
        -------
            action_size : dimension of each action
            buffer_size : maximum size of buffer
            batch_size : size of each training batch
            seed : random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                               "action",
                                                               "reward",
                                                               "next_state",
                                                               "done"])
        self.seed = random.seed(seed)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def add(self,state, action, reward, next_state,done):
        """Add a new experience to memory"""
        e = self.experiences(state,action,reward,next_state,done)
        self.memory.append(e)
        
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory,k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        return (states,actions,rewards,next_states,dones)

    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)