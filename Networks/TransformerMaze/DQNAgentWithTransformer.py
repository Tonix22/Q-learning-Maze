import torch
import torch.nn as nn
from Networks.TransformerMaze.TransformerDQN import TransformerDQN
import random
import numpy as np

class DQNAgentWithTransformer:
    def __init__(self, state_dim, num_actions, seq_length, hidden_dim=64, nhead=4, num_layers=2, 
                 gamma=0.99, max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.001, learning_rate=0.001,
                 memory_size=100, batch_size=1, update_target_freq=100):
        self.state_dim = state_dim          # e.g., if using one-hot states, state_dim = maze_size*maze_size
        self.num_actions = num_actions
        self.seq_length = seq_length
        
        self.gamma = gamma
        self.epsilon = max_epsilon
        self.epsilon_min = min_epsilon
        self.epsilon_decay = decay_rate
        self.learning_rate = learning_rate
        
        from collections import deque
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        self.steps_done = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Transformer-based network
        self.policy_net = TransformerDQN(input_dim=state_dim, hidden_dim=hidden_dim, nhead=nhead, 
                                          num_layers=num_layers, seq_length=seq_length, num_actions=num_actions).to(self.device)
        self.target_net = TransformerDQN(input_dim=state_dim, hidden_dim=hidden_dim, nhead=nhead, 
                                          num_layers=num_layers, seq_length=seq_length, num_actions=num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
    
    def get_action(self, state_sequence):
        """
        Args:
            state_sequence: numpy array of shape [seq_length, state_dim]
        Returns:
            action: integer index of chosen action.
        """
        # Convert to tensor and add batch dimension
        state_tensor = torch.FloatTensor(state_sequence).unsqueeze(0).to(self.device)  # shape: [1, seq_length, state_dim]
        if random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            action = torch.argmax(q_values).item()
        else:
            action = random.randrange(self.num_actions)
        return action
    
    def store_transition(self, state_sequence, action, reward, next_state_sequence, done):
        self.memory.append((state_sequence, action, reward, next_state_sequence, done))
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([m[0] for m in minibatch])).to(self.device)           # [B, seq_length, state_dim]
        actions = torch.LongTensor(np.array([m[1] for m in minibatch])).unsqueeze(1).to(self.device)  # [B, 1]
        rewards = torch.FloatTensor(np.array([m[2] for m in minibatch])).unsqueeze(1).to(self.device) # [B, 1]
        next_states = torch.FloatTensor(np.array([m[3] for m in minibatch])).to(self.device)         # [B, seq_length, state_dim]
        dones = torch.FloatTensor(np.array([m[4] for m in minibatch])).unsqueeze(1).to(self.device)    # [B, 1]
        
        # Compute current Q-values for taken actions
        current_q = self.policy_net(states).gather(1, actions)
        
        # Compute next Q-values from target network (max over actions)
        next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = nn.MSELoss()(current_q, target_q.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= np.exp(-self.epsilon_decay)
        
        self.steps_done += 1
        if self.steps_done % self.update_target_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
