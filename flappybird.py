import pygame
import sys
import random
import numpy as np
import json
import matplotlib.pyplot as plt


# Import your Q-Learning agent for Flappy Bird.
from Networks.QlearningFlappy import QLearningFlappy
# Import your drawing/rendering helper for Flappy Bird.
from Game.renderflapy import DrawGraphics

# -------------------------
# Load configuration from JSON.
# -------------------------
config_file = 'config/flappyconfig.json'
with open(config_file, 'r') as file:
    data = json.load(file)

fps           = data['fps']
max_steps     = data['max_steps']
max_episodes  = data['max_episodes']
load_weights  = data['load_weights']
enable_render = data['enable_render']

WIDTH         = data.get('width', 400)
HEIGHT        = data.get('height', 600)
GRAVITY       = data.get('gravity', 0.8)
JUMP_STRENGTH = data.get('jump_strength', -10)
PIPE_WIDTH    = data.get('pipe_width', 70)
PIPE_GAP      = data.get('pipe_gap', 200)
PIPE_VELOCITY = data.get('pipe_velocity', 3)
PIPE_FREQ     = data.get('pipe_frequency', 3000)  # in milliseconds

BIRD_X        = 50
BIRD_RADIUS   = 20

# -------------------------
# GAME_FLAPPY class definition.
# -------------------------
class GAME_FLAPPY():
    def __init__(self, load_weights=False, enable_render=True):
        # Internal variables.
        self.run = True
        self.episode = 0
        self.steps = 0
        self.reward = 0
        self.epsilon = 0
        self.action = None
        self.succes_history = []
        self.success_episode = []
        self.enable_render = enable_render
        self.record_steps = 0 

        # Initialize pygame.
        pygame.init()
        self.width = WIDTH
        self.height = HEIGHT

        # Set up rendering if enabled.
        if self.enable_render:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Flappy Bird Q-Learning")
            # Create the rendering engine.
            self.draw = DrawGraphics(self.width, self.height)
            self.clock = self.draw.getclock()  # Assumes your DrawGraphics provides a clock.
        else:
            self.screen = None

        # Initialize bird parameters.
        self.bird_x = BIRD_X
        self.bird_y = self.height // 2
        self.bird_radius = BIRD_RADIUS
        self.bird_velocity = 0

        # Pipe parameters.
        self.pipes = []
        self.last_pipe_time = pygame.time.get_ticks() - PIPE_FREQ

        # Q-Learning agent.
        # We discretize the state space into 10 bins for the bird's y-position and for the pipe gap.
        bins_y = 10
        bins_pipe = 10
        self.ql = QLearningFlappy(bins_y, bins_pipe, self.height)

        if load_weights:
            self.ql.q_table = np.load('Checkpoint/Q_table_flappy.npy')
            self.ql.max_epsilon = 0
            print("load weights sucessfuly")
        # Initialize state using the current bird_y and a default pipe gap (middle of the screen).
        self.state = self.ql.get_state(self.bird_y, self.height // 2)

    def load_window(self):
        # Enforce FPS, process events, and update the window.
        self.clock.tick(fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run = False
        self.draw.window()

    def load_epsilon(self):
        # Calculate epsilon and choose an action from the Q-Learning agent.
        self.epsilon = self.ql.get_epsilon(self.episode)
        current_state = self.get_state()
        self.action = self.ql.choose_action(current_state, self.epsilon)

    def get_nearest_pipe(self):
        # Returns the nearest pipe ahead of the bird.
        nearest = None
        min_distance = float('inf')
        for pipe in self.pipes:
            distance = (pipe['x'] + PIPE_WIDTH) - self.bird_x
            if distance > 0 and distance < min_distance:
                min_distance = distance
                nearest = pipe
        if nearest is None:
            nearest = {'gap_y': self.height // 2}
        return nearest

    def get_state(self):
        # Discretizes the state based on the bird's y-position and the nearest pipe's gap.
        nearest_pipe = self.get_nearest_pipe()
        return self.ql.get_state(self.bird_y, nearest_pipe['gap_y'])

    def update_action(self):
        # Save the previous position (if needed) and execute the chosen action.
        self.y_before = self.bird_y
        if self.action == 'flap':
            self.bird_velocity = JUMP_STRENGTH
        # If the action is "do_nothing", gravity will act normally.

    def increment_step(self):
        self.steps += 1
        if self.enable_render:
            self.draw.clear()  # Clear the screen.
            self.draw.draw_pipes(self.pipes, PIPE_WIDTH, PIPE_GAP, self.height)
            self.draw.draw_bird(self.bird_x, self.bird_y, self.bird_radius)
            self.draw.draw_score(self.steps)  # Display the step count (or score).
            self.draw.display_flip()

    def update_physics(self):
        # Update the bird's velocity and position.
        self.bird_velocity += GRAVITY
        self.bird_y += self.bird_velocity

    def update_pipes(self):
        # Generate new pipes at fixed intervals.
        current_time = pygame.time.get_ticks()
        if current_time - self.last_pipe_time > PIPE_FREQ:
            gap_y = random.randint(PIPE_GAP // 2 + 10, self.height - PIPE_GAP // 2 - 10)
            self.pipes.append({'x': self.width, 'gap_y': gap_y, 'passed': False})
            self.last_pipe_time = current_time

        # Move pipes to the left.
        for pipe in self.pipes:
            pipe['x'] -= PIPE_VELOCITY
        self.pipes = [pipe for pipe in self.pipes if pipe['x'] + PIPE_WIDTH > 0]

    def reward_select(self):
        # Assign a small positive reward for survival.
        reward = 0.1
        # Check collision with the top or bottom of the screen.
        if self.bird_y - self.bird_radius < 0 or self.bird_y + self.bird_radius > self.height:
            reward = -1
        # Check collision with pipes.
        for pipe in self.pipes:
            if (self.bird_x + self.bird_radius > pipe['x'] and 
                self.bird_x - self.bird_radius < pipe['x'] + PIPE_WIDTH):
                if (self.bird_y - self.bird_radius < pipe['gap_y'] - PIPE_GAP // 2 or 
                    self.bird_y + self.bird_radius > pipe['gap_y'] + PIPE_GAP // 2):
                    reward = -1
                    break
        self.reward = reward

    def check_step_limit(self):
        if self.steps > max_steps:
            self.reset_episode()

    def reset_episode(self):
        # Record the number of steps survived for this episode and reset variables.
        
        if(self.record_steps < self.steps):
            self.record_steps = self.steps
            print(f"Episode : {self.episode} new record steps: {self.steps}")
            self.succes_history.append(self.steps)
            self.success_episode.append(self.episode)
        
        self.steps = 0
        self.bird_y = self.height // 2
        self.bird_velocity = 0
        self.pipes = []
        self.last_pipe_time = pygame.time.get_ticks() - PIPE_FREQ
        self.episode += 1

    def main_loop(self):
        while self.run and self.episode < max_episodes:
            if self.enable_render:
                self.load_window()

            self.load_epsilon()
            self.update_action()
            self.update_physics()
            self.update_pipes()
            self.increment_step()
            self.reward_select()

            # If collision occurs (reward == -1), update Q-table and reset the episode.
            if self.reward == -1:
                next_state = self.get_state()
                self.ql.update_q_value(self.state, self.action, self.reward, next_state)
                self.reset_episode()
                random.seed(42)
                continue

            # Optionally, reward passing a pipe (only once per pipe).
            for pipe in self.pipes:
                if not pipe.get('passed') and pipe['x'] + PIPE_WIDTH < self.bird_x:
                    pipe['passed'] = True
                    self.reward = 1

            next_state = self.get_state()
            self.ql.update_q_value(self.state, self.action, self.reward, next_state)
            self.state = next_state
            self.check_step_limit()

        if self.enable_render:
            self.draw.quit()
        np.save('Checkpoint/Q_table_flappy.npy', self.ql.q_table)
        #print(self.ql.q_table)
        print(f'Episodes: {self.episode}')
        plt.plot(np.array(self.success_episode),np.array(self.succes_history))
        plt.title("Steps Survived per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.show()
        sys.exit()

if __name__ == "__main__":
    random.seed(42)
    game = GAME_FLAPPY(load_weights, enable_render)
    game.main_loop()
