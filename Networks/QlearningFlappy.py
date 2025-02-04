import random
import numpy as np
import json

# Load Q-learning configuration parameters from a JSON file
file_name = 'config/qlearning_flappy.json'
with open(file_name, 'r') as file:
    data = json.load(file)
    
max_epsilon = data['max_epsilon']  # maximum exploration rate
min_epsilon = data['min_epsilon']  # minimum exploration rate
decay_rate  = data['decay_rate']   # decay rate for epsilon
alpha       = data['alpha']        # learning rate
gamma       = data['gamma']        # discount factor

# Define the action space: "flap" to jump, "do_nothing" to let gravity act
actions = ['flap', 'do_nothing']

class QLearningFlappy:
    def __init__(self, bins_y, bins_pipe, screen_height):
        """
        Initializes the QLearningFlappy agent.

        Parameters:
        - bins_y (int): Number of bins to discretize the bird's vertical position.
        - bins_pipe (int): Number of bins to discretize the pipe gap vertical position.
        - screen_height (int): The height of the game screen (for discretization).
        """
        self.bins_y = bins_y
        self.bins_pipe = bins_pipe
        self.screen_height = screen_height
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        # The Q-table is a 3D array: [bird_y_bin, pipe_gap_bin, action_index]
        self.q_table = np.zeros((bins_y, bins_pipe, len(actions)))
    
    def get_state(self, bird_y, pipe_gap_y):
        """
        Discretizes the continuous state into a tuple of indices.

        Parameters:
        - bird_y (float): The bird's vertical position.
        - pipe_gap_y (float): The vertical position of the pipe gap.

        Returns:
        - state (tuple): A tuple (bird_y_bin, pipe_gap_bin)
        """
        # Compute which bin the bird's y-coordinate falls into.
        bird_y_bin = min(int(bird_y / (self.screen_height / self.bins_y)), self.bins_y - 1)
        # Compute which bin the pipe gap's y-coordinate falls into.
        pipe_gap_bin = min(int(pipe_gap_y / (self.screen_height / self.bins_pipe)), self.bins_pipe - 1)
        return (bird_y_bin, pipe_gap_bin)
    
    def get_epsilon(self, episode):
        """
        Calculates the exploration rate for a given episode using exponential decay.
        """
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-decay_rate * episode)
    
    def choose_action(self, state, epsilon):
        """
        Chooses an action using an epsilon-greedy policy.

        Parameters:
        - state (tuple): The current discretized state.
        - epsilon (float): The current exploration rate.

        Returns:
        - action (str): The selected action.
        """
        if random.random() > epsilon:
            # Exploitation: choose the best known action for this state.
            bird_y_bin, pipe_gap_bin = state
            action_idx = np.argmax(self.q_table[bird_y_bin, pipe_gap_bin])
        else:
            # Exploration: randomly choose an action.
            action_idx = random.randint(0, len(actions)-1)
        return actions[action_idx]
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Updates the Q-table based on the agent's experience.

        Parameters:
        - state (tuple): The current state (bird_y_bin, pipe_gap_bin).
        - action (str): The action taken.
        - reward (float): The reward received.
        - next_state (tuple): The state after taking the action.
        """
        bird_y_bin, pipe_gap_bin = state
        next_bird_y_bin, next_pipe_gap_bin = next_state
        action_index = actions.index(action)
        # Q-learning update rule.
        best_next_q = np.max(self.q_table[next_bird_y_bin, next_pipe_gap_bin])
        self.q_table[bird_y_bin, pipe_gap_bin, action_index] += alpha * (reward + gamma * best_next_q - self.q_table[bird_y_bin, pipe_gap_bin, action_index])
