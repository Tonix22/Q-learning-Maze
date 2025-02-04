import numpy as np
from MazeGenerator.KruskalMaze import RectangularKruskalMaze
from Game.render import DrawGraphics
from Networks.QlearningMaze import QlearningMaze
import json
import matplotlib.pyplot as plt

file_name = 'config/mazeconfig.json'
with open(file_name, 'r') as file:
    data = json.load(file)

#constants
MIDDLE_X   = 25
MIDDLE_Y   = 25
BLOCK_SIZE = 50
WALL       = 0
GOAL       = 2
STEP       = 50

#json parameters
fps          = data['fps']
max_steps    = data['max_steps']
max_episodes = data['max_episodes']
size         = data['size'] # odd number only
load_weights = data['load_weights']
enable_render = data['enable_render']

maze_param = (size-1)//2
width  = size * STEP
height = size * STEP

class GAME():
    def __init__(self,load_weights=False,enable_render=True):
        #internal variables
        self.run     = True
        self.episode = 0
        self.state   = 0
        self.steps   = 0
        self.wins    = 0
        self.reward  = 0
        self.epsilon = 0
        self.action  = None
        self.succes_history = []
        # Add a flag to control rendering
        self.enable_render = enable_render
        
        # Helper classes
        # Create realistic Maze
        algo = RectangularKruskalMaze(maze_param, maze_param)
        self.grid = algo.create_maze_array() #return as np
        
        #Q-Learning class
        self.ql   = QlearningMaze(size)
        
        if(load_weights == True):
            self.ql.q_table = np.load('Checkpoint/Q_table.npy')
            self.grid       = np.load('Checkpoint/Maze.npy')
            self.ql.max_epsilon = self.ql.min_epsilon
            
        else:
            #save maze for future recovery
            np.save('Checkpoint/Maze.npy',self.grid)

        #pygame graphics engine
        if self.enable_render:
            self.draw = DrawGraphics(self.grid,width,height)

        #init x y coordinates
        self.x_pos = MIDDLE_X # current ball center window
        self.y_pos = MIDDLE_Y+BLOCK_SIZE
    
        if self.enable_render:
             #get systic clock
            self.clock   = self.draw.getclock()
            #first draw circle
            self.draw.circle(self.x_pos,self.y_pos)

    def load_window(self):
        #systick
        self.clock.tick(fps)
        
        #window title
        self.draw.windowtitle(self.episode,self.steps)
        
        #close window game
        self.run = self.draw.Isquit()
        
        #refresh draw
        self.draw.window()
    
    # uses Q-Learning class
    def load_epsilon(self):
        #calculate epsilon
        self.epsilon = self.ql.get_epsilon(self.episode)
            
        if self.episode%10 == 0 and self.steps == 0:
            print(self.epsilon)
        
        self.action = self.ql.take_action(self.state, self.epsilon)

    def update_action(self):
        self.x_before = self.x_pos
        self.y_before = self.y_pos
        
        if self.action=='left' and (self.x_pos > MIDDLE_X):
            self.x_pos -= BLOCK_SIZE
        elif self.action=='right' and (self.x_pos < width-MIDDLE_X) :
            self.x_pos += BLOCK_SIZE
        elif self.action=='up' and (self.y_pos > MIDDLE_Y) :
            self.y_pos -= BLOCK_SIZE
        elif self.action=='down' and (self.y_pos < height-MIDDLE_Y):
            self.y_pos += BLOCK_SIZE
    
    def increment_step(self):
        self.steps += 1
        if self.enable_render:
            self.draw.circle(self.x_pos, self.y_pos)
            self.draw.display_flip()
        
    def reward_select(self):
        #collision make wall return to step before
        if self.grid[self.y_pos//STEP][self.x_pos//STEP] == WALL:
            self.x_pos = self.x_before
            self.y_pos = self.y_before
            self.reward   = -1 # punish the q-learning
        #you reach the goal
        elif self.grid[self.y_pos//STEP][self.x_pos//STEP] == GOAL:
            #reset game to init pos
            self.x_pos = MIDDLE_X
            self.y_pos = MIDDLE_Y+BLOCK_SIZE
            self.episode += 1
            self.reward   = 1 # positive reward
            self.wins    += 1
            print("WIN!!!")
            print(f"Episode {self.episode} finished after {self.steps} steps")
            self.succes_history.append(self.steps)
            self.steps = 0
            
        else:
            self.reward = 0
            
    def check_step_limit(self):
        if self.steps > max_steps:
            self.x_pos = MIDDLE_X
            self.y_pos = MIDDLE_Y+BLOCK_SIZE
            self.episode += 1
            self.steps    = 0
            print(f"Episode {self.episode} finished after {max_steps} steps")

    def main_loop(self):
        while self.run & (self.episode < max_episodes):
            
            if self.enable_render:
                self.load_window()
            
            self.load_epsilon()
            self.update_action()
            self.increment_step()
            self.reward_select()
            self.check_step_limit()
            
            #calculate next step
            next_state = size * (self.y_pos // STEP) + self.x_pos // STEP
            #update Q-table
            self.ql.update_q_table(self.state, self.action, self.reward, next_state)
            #check next-state
            self.state = next_state
                            
        if self.enable_render:
            self.draw.QuitGame()
        #save trainning
        np.save('Checkpoint/Q_table.npy',self.ql.q_table)
        print(self.ql.q_table)
        print(f'Wins {self.wins}')
        
        my_array = np.array(self.succes_history)
        # Plotting the array
        plt.plot(my_array)
        plt.title('My Array Plot')
        plt.xlabel('Index')
        plt.ylabel('Steps')
        plt.show()

if __name__ == "__main__":
    maze = GAME(load_weights,enable_render)
    maze.main_loop()