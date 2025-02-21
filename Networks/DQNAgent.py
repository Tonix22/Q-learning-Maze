import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import json


class SimpleDQN(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=64):
        super(SimpleDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, num_actions, gamma=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, learning_rate=1e-3, memory_size=100, batch_size=10):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        print(f"Gamma: {gamma}")
        print(f"epsilon: {epsilon}")
        print(f"epsilon_min: {epsilon_min}")
        print(f"epsilon_decay: {epsilon_decay}")
        print(f"learning_rate: {learning_rate}")
        print(f"memory_size: {memory_size}")
        print(f"batch_size: {batch_size}")
        
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = SimpleDQN(state_dim, num_actions).to(self.device)
        self.target_net = SimpleDQN(state_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.steps_done = 0
    
    def get_action(self, state):
        """
        Args:
            state: np.array of shape (state_dim,)
        Returns:
            action: integer index of chosen action.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return torch.argmax(q_values).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([m[0] for m in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([m[1] for m in minibatch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([m[2] for m in minibatch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([m[3] for m in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([m[4] for m in minibatch])).unsqueeze(1).to(self.device)
        
        current_q = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = nn.MSELoss()(current_q, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(-self.epsilon_decay * self.steps_done)
            #print(f"self.epsilon: {self.epsilon}")
        else:
            print("MIN EPSILON REACHED")
            
        self.steps_done += 1
        if self.steps_done % 1000 == 0:
            print(f"step:{self.steps_done}, loss:{loss}")
            self.target_net.load_state_dict(self.policy_net.state_dict())
