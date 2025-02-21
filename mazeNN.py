import numpy as np
from MazeGenerator.KruskalMaze import RectangularKruskalMaze
from Game.render import DrawGraphics
from Networks.DQNAgent import DQNAgent  # Updated import
import json
import matplotlib.pyplot as plt
import torch

file_name = 'config/mazeconfig.json'
with open(file_name, 'r') as file:
    data = json.load(file)

# constants
MIDDLE_X   = 25
MIDDLE_Y   = 25
BLOCK_SIZE = 50
WALL       = 0
GOAL       = 2
STEP       = 50

# json parameters
fps          = data['fps']
max_steps    = data['max_steps']
max_episodes = data['max_episodes']
size         = data['size']  # odd number only
load_weights = data['load_weights']
enable_render = data['enable_render']

file_name = 'config/qlearning.json'
with open(file_name, 'r') as file:
    data = json.load(file)
    
max_epsilon = data['max_epsilon']
min_epsilon = data['min_epsilon']
decay_rate  = data['decay_rate']
alpha       = data['alpha']
gamma       = data['gamma']
learningrate = data['lr']
memorysize = data['mem_size']
batchsize = data['batchsize']

maze_param = (size - 1) // 2
width  = size * STEP
height = size * STEP

class GAME():
    def __init__(self, load_weights=False, enable_render=True):
        # internal variables
        self.run = True
        self.episode = 0
        self.state = 0  # state is represented as an index (will be one-hot encoded)
        self.steps = 0
        self.wins = 0
        self.reward = 0
        self.action = None
        self.succes_history = []
        self.enable_render = enable_render
        
        # Create realistic Maze
        algo = RectangularKruskalMaze(maze_param, maze_param)
        self.grid = algo.create_maze_array()  # returns np.array
        
        # DQN Agent using a simple feed-forward network
        state_dim = size * size  # one-hot representation length
        num_actions = 4
        self.agent = DQNAgent(state_dim=state_dim, 
                              num_actions=num_actions,
                              gamma=gamma,
                              epsilon=max_epsilon,
                              epsilon_min=min_epsilon,
                              epsilon_decay=decay_rate,
                              learning_rate=learningrate,
                              memory_size=memorysize,
                              batch_size=batchsize)
        
        if load_weights:
            checkpoint = torch.load('Checkpoint/agent_checkpoint.pth')
            self.agent.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.agent.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.grid = np.load('Checkpoint/Maze.npy')
            self.agent.epsilon = 0.1
            print("Loaded agent checkpoint and maze from Checkpoint!")
        else:
            # save maze for future recovery
            np.save('Checkpoint/Maze.npy', self.grid)
        
        # pygame graphics engine
        if self.enable_render:
            self.draw = DrawGraphics(self.grid, width, height)
        
        # initial x,y coordinates
        self.x_pos = MIDDLE_X
        self.y_pos = MIDDLE_Y + BLOCK_SIZE
        self.x_before = self.x_pos  # used to revert on collision
        self.y_before = self.y_pos
        
        if self.enable_render:
            self.clock = self.draw.getclock()
            self.draw.circle(self.x_pos, self.y_pos)
    
    def load_window(self):
        self.clock.tick(fps)
        self.draw.windowtitle(self.episode, self.steps)
        self.run = self.draw.Isquit()
        self.draw.window()
    
    def load_action(self):
        # Build one-hot state representation from the current state index.
        state_onehot = np.zeros(size * size)
        state_onehot[self.state] = 1
        self.action = self.agent.get_action(state_onehot)
    
    def update_action(self):
        self.x_before = self.x_pos
        self.y_before = self.y_pos
        
        # Map integer action to a movement direction.
        actions = ['up', 'down', 'left', 'right']
        action_str = actions[self.action]
        
        if action_str == 'left' and (self.x_pos > MIDDLE_X):
            self.x_pos -= BLOCK_SIZE
        elif action_str == 'right' and (self.x_pos < width - MIDDLE_X):
            self.x_pos += BLOCK_SIZE
        elif action_str == 'up' and (self.y_pos > MIDDLE_Y):
            self.y_pos -= BLOCK_SIZE
        elif action_str == 'down' and (self.y_pos < height - MIDDLE_Y):
            self.y_pos += BLOCK_SIZE
    
    def increment_step(self):
        self.steps += 1
        if self.enable_render:
            self.draw.circle(self.x_pos, self.y_pos)
            self.draw.display_flip()
    
    def reward_select(self):
        # Collision with wall: revert to previous position and penalize.
        if self.grid[self.y_pos // STEP][self.x_pos // STEP] == WALL:
            self.x_pos = self.x_before
            self.y_pos = self.y_before
            self.reward = -1
        elif self.x_pos == self.x_before and self.y_pos == self.y_before:
            self.reward = -1
        # Reaching the goal: reset position, reward, and count a win.
        elif self.grid[self.y_pos // STEP][self.x_pos // STEP] == GOAL:
            self.x_pos = MIDDLE_X
            self.y_pos = MIDDLE_Y + BLOCK_SIZE
            self.episode += 1
            self.reward = 1
            self.wins += 1
            print("WIN!!!")
            print(f"Episode {self.episode} finished after {self.steps} steps")
            self.succes_history.append(self.steps)
            self.steps = 0
        else:
            self.reward = 0
            
    def check_step_limit(self):
        if self.steps > max_steps:
            self.x_pos = MIDDLE_X
            self.y_pos = MIDDLE_Y + BLOCK_SIZE
            self.episode += 1
            self.steps = 0
            print(f"Episode {self.episode} finished after {max_steps} steps")
    
    def main_loop(self):
        while self.run and (self.episode < max_episodes):
            if self.enable_render:
                self.load_window()
            
            self.load_action()
            self.update_action()
            self.increment_step()
            self.reward_select()
            self.check_step_limit()
            
            # Compute next state index and build one-hot representation.
            next_state_index = size * (self.y_pos // STEP) + self.x_pos // STEP
            next_state = np.zeros(size * size)
            next_state[next_state_index] = 1
            
            # Store transition in replay memory.
            current_state = np.zeros(size * size)
            current_state[self.state] = 1
            done = 0  # Using 0 to indicate that the episode is not terminal.
            self.agent.store_transition(current_state, self.action, self.reward, next_state, done)
            self.agent.replay()
            
            # Update current state index.
            self.state = next_state_index
        
        if self.enable_render:
            self.draw.QuitGame()
        
        # Save training checkpoint.
        checkpoint = {
            'policy_net_state_dict': self.agent.policy_net.state_dict(),
            'target_net_state_dict': self.agent.target_net.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'episode': self.episode,
            'succes_history': self.succes_history,
        }
        
        if load_weights == False:
            torch.save(checkpoint, 'Checkpoint/agent_checkpoint.pth')
            print("Checkpoint saved to Checkpoint/agent_checkpoint.pth")
            
        print(f'Wins {self.wins}')
        
        my_array = np.array(self.succes_history)
        plt.plot(my_array)
        plt.title('My Array Plot')
        plt.xlabel('Index')
        plt.ylabel('Steps')
        plt.show()

if __name__ == "__main__":
    maze = GAME(load_weights, enable_render)
    maze.main_loop()
