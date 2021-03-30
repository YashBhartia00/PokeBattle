import numpy as np
import random 
from collections import namedtuple, deque 

##Importing the model 
# import QNetwork, ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque 
import random


class Agent:
    """
    Agent class that interacts with the environment and learns

    Hyperparameters
    -------

    BUFFER_SIZE : replay buffer size
    BATCH_SIZE : minibatch size
    GAMMA : discount factor
    TAU : for soft update of target parameters
    LR : learning rate
    UPDATE_EVERY : how often to update the network

    """

    def __init__(self, game, state_size=59 , action_size=6, seed=0):
        """
        Initializes the playing agent

        game = GameManager object
        """
        self.BUFFER_SIZE = int(1e5)  #replay buffer size
        self.BATCH_SIZE = 128         # minibatch size
        self.GAMMA = 0.99            # discount factor
        self.TAU = 1e-4              # for soft update of target parameters
        self.LR = 5e-5               # learning rate
        self.UPDATE_EVERY = 5        # how often to update the network

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.game = game
        self.team = game.team
        self.opponent = game.opp_team

        # The array of all the agents moves
        self.moves = np.array(
            [
                [self.game.moves[self.team[i][j + 9]] for j in range(4)]
                for i in range(len(self.team))
            ]
        )

        # The index of the current pokemon is maintained by the GameManager
        self.current_pokemon = self.team[self.game.index]

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        #Q- Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
        
        #optimizer
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=self.LR)

        # Replay memory 
        self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE,self.BATCH_SIZE,seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def updateAgent(self, game ):
        self.game = game
        self.team = game.team
        self.opponent = game.opp_team 
        self.moves = np.array(
            [
                [self.game.moves[self.team[i][j + 9]] for j in range(4)]
                for i in range(len(self.team))
            ]
        )

        # The index of the current pokemon is maintained by the GameManager
        self.current_pokemon = self.team[self.game.index]

    def soft_update(self, local_model, target_model, tau):
        """update the target network"""

        for target_param, local_param in zip(target_model.parameters(),
                                            local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
            

    def step(self, state, action, reward, next_state, done):
        """
        updates every step, decided weather to fill replay buffer or to learn
        learns if replay buffer is full

        parameter
        -------
        state : state of the environment
        action : action taken 
        reward : current reward
        next_state : state of environment after action 
        done : True if current game is completed
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step+1)% self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get radom subset and learn

            if len(self.memory)>self.BATCH_SIZE:
                experience = self.memory.sample()
                self.update(experience, self.GAMMA)


    def select_action(self, state, eps = 0):
        """
        selects the action according to state
        
        Parameters
        ------
        state : the state array 
        eps : value of epsilon

        Returns
        -------
        action : index of action to be taken 
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        #Epsilon - greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        action = self.moves[self.game.index].T[2].argmax()
        return action

    def update(self, experiences, gamma):

        states, actions, rewards, next_state, dones = experiences

        criterion = torch.nn.MSELoss()

        self.qnetwork_local.train()
        self.qnetwork_target.eval()

        #shape of output from the model (batch_size,action_dim) = (64,6)
        predicted_targets = self.qnetwork_local(states).gather(1,actions)
    
        with torch.no_grad():
            labels_next = self.qnetwork_target(next_state).detach().max(1)[0].unsqueeze(1)

        #computing the loss minimizing it
        labels = rewards + (gamma* labels_next*(1-dones))
        loss = criterion(predicted_targets,labels).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.soft_update(self.qnetwork_local,self.qnetwork_target,self.TAU)

    def learn(self, num_epochs=20000, episode_len=1000, eps_start = 1.0, eps_end = 0.01, eps_decay = 0.9996):
        """

        Arguments
        -------
        num_epochs (int): Number of epochs to train the agent for
        episode_len (int): Number of timesteps (not including opponent timesteps) per pokebattle
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon 
        eps_decay (float): mutiplicative factor for decreasing epsilon

        """
        winTotal = [] # list containing wins
        eps = eps_start
        wins =0
        scores_window = deque(maxlen=100)
        printNum = 100
        for epoch in range(1, num_epochs+1):
            env = self.game
            state = env.reset()
            score = 0

            for i_episode in range(episode_len):
                action = self.select_action(state, eps)
                next_state,reward,done,win = env.step(action)
                self.step(state,action,reward,next_state,done)
                # next_state, reward, done, info = self.game.step(action)

                # episode_reward += reward
                state = next_state
                score += reward
                # self.update()
                if done:
                    wins = wins+1 if win else wins
                    break
                
                scores_window.append(score)
                eps = max(eps*eps_decay,eps_end)
                # state = next_state

            if epoch %printNum==0:
                print('\rEpisode {}\tAverage Score {:.2f}\tWin Rate: {}%'.format(epoch,np.mean(scores_window), 100*wins/printNum))
                winTotal.append(wins)
                wins =0

            if 100*wins/printNum >=80.0 :
                print('\nEnvironment solve in {:d} epsiodes!\tAverage score: {:.2f}'.format(epoch-100,
                                                                                        np.mean(scores_window)))
                torch.save(self.qnetwork_local.state_dict(),'checkpoint.pth')
                break

        return winTotal

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
