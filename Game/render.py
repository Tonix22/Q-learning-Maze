import pygame

class DrawGraphics():
    def __init__(self, grid,width,height):
        self.grid = grid
        self.win = pygame.display.set_mode((width, height))

    def window(self):
        self.win.fill((0,0,0))
        self.rectangles()
        
    #red,green,blue
    def rectangles(self):
        size = len(self.grid)
        for i in range(size):
            for j in range(size):
                if self.grid[j][i] == 1:
                    color = (255, 255, 255)
                elif self.grid[j][i] == 0:
                    color = (255, 0, 0)
                elif self.grid[j][i] >= 2:
                    color = (0, 255, 0)
                pygame.draw.rect(self.win, color, (i*50, j*50, 48, 48), 0)
    
    def circle(self,x_pos,y_pos):
        pygame.draw.circle(self.win, (0, 0, 255), (x_pos, y_pos), 20, 0)
    
    def windowtitle(self,episode,step):
        pygame.display.set_caption(f"Maze Game - Episode {episode}, step {step}")
    
    def Isquit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True
    def QuitGame(self):
        pygame.quit()
    
    def getclock(self):
        return pygame.time.Clock()
    
    def display_flip(self):
        pygame.display.flip()