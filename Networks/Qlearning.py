import random
import numpy as np
import json

file_name = 'config/qlearning.json'
with open(file_name, 'r') as file:
    data = json.load(file)
    
max_epsilon = data['max_epsilon']
min_epsilon = data['min_epsilon']
decay_rate  = data['decay_rate']
alpha       = data['alpha']
gamma       = data['gamma']

actions = ['up', 'down', 'left', 'right']

class Qlearning():
    def __init__(self, size):
        self.size = size
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        #four posibilities manhatan distance
        self.q_table = np.zeros((size*size,4))
    
    def get_epsilon(self,episode):
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-decay_rate * episode)

    def take_action(self,state, epsilon):
        if random.random() > epsilon:
            return actions[np.argmax(self.q_table[state])]
        else:
            possible_actions = []
            if state < self.size*(self.size-1):
                possible_actions.append('down')
            if state >= self.size:
                possible_actions.append('up')
            if state % self.size != 0:
                possible_actions.append('left')
            if state % self.size != self.size-1:
                possible_actions.append('right')
            return random.choice(possible_actions)
    
    def update_q_table(self,state, action, reward, next_state):  
        self.q_table[state][actions.index(action)] += alpha * (reward + gamma * np.max(self.q_table[next_state]) - self.q_table[state][actions.index(action)])