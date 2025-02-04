import pygame

class DrawGraphics:
    def __init__(self, width, height):
        """
        Initializes the drawing engine for Flappy Bird.

        Parameters:
            width (int): Width of the game window.
            height (int): Height of the game window.
        """
        self.width = width
        self.height = height
        # Get the display surface (assumes pygame.display.set_mode has already been called)
        self.screen = pygame.display.get_surface()
        # Background color (sky blue)
        self.background_color = (135, 206, 235)
        # Font for drawing text (score, etc.)
        self.font = pygame.font.SysFont("Arial", 32)

    def getclock(self):
        """
        Returns a new pygame clock to manage the frame rate.
        """
        return pygame.time.Clock()

    def clear(self):
        """
        Clears the screen by filling it with the background color.
        """
        self.screen.fill(self.background_color)

    def draw_pipes(self, pipes, pipe_width, pipe_gap, screen_height):
        """
        Draws all pipes on the screen.

        Parameters:
            pipes (list): A list of dictionaries. Each dictionary should have:
                          - 'x': The x-coordinate of the pipe.
                          - 'gap_y': The vertical center of the gap.
            pipe_width (int): The width of each pipe.
            pipe_gap (int): The vertical gap between the top and bottom pipes.
            screen_height (int): The height of the game screen.
        """
        GREEN = (0, 200, 0)
        for pipe in pipes:
            # Draw top pipe: from the top to just before the gap.
            top_rect = pygame.Rect(pipe['x'], 0, pipe_width, pipe['gap_y'] - pipe_gap // 2)
            pygame.draw.rect(self.screen, GREEN, top_rect)
            # Draw bottom pipe: from just after the gap to the bottom.
            bottom_rect = pygame.Rect(pipe['x'], pipe['gap_y'] + pipe_gap // 2, 
                                      pipe_width, screen_height - (pipe['gap_y'] + pipe_gap // 2))
            pygame.draw.rect(self.screen, GREEN, bottom_rect)

    def draw_bird(self, bird_x, bird_y, bird_radius):
        """
        Draws the bird as a circle on the screen.

        Parameters:
            bird_x (float): The x-coordinate of the bird's center.
            bird_y (float): The y-coordinate of the bird's center.
            bird_radius (int): The radius of the bird.
        """
        WHITE = (255, 255, 255)
        pygame.draw.circle(self.screen, WHITE, (int(bird_x), int(bird_y)), bird_radius)

    def draw_score(self, score):
        """
        Draws the current score (or step count) on the screen.

        Parameters:
            score (int): The current score or number of steps survived.
        """
        WHITE = (255, 255, 255)
        score_surface = self.font.render(f"Score: {score}", True, WHITE)
        # Center the score horizontally near the top of the window.
        self.screen.blit(score_surface, (self.width // 2 - score_surface.get_width() // 2, 20))

    def display_flip(self):
        """
        Updates the display to reflect all drawing calls.
        """
        pygame.display.flip()

    def window(self):
        """
        Performs a display update. (This method can be used if additional handling is needed.)
        """
        pygame.display.update()

    def windowtitle(self, episode, steps):
        """
        Updates the window caption with the current episode and steps.

        Parameters:
            episode (int): The current episode number.
            steps (int): The number of steps in the current episode.
        """
        pygame.display.set_caption(f"Flappy Bird Q-Learning - Episode: {episode}, Steps: {steps}")

    def quit(self):
        """
        Properly quits the pygame window.
        """
        pygame.quit()
