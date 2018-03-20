'''
Author: Uriel Sade
Date: June 30th 2017

Snake Game controlled by a neural network with an API similar to OpenAI Gym
( step(a), sample(), reset() )
'''

#TODO: Documentation

import pygame
import sys
import random
import numpy as np
from game_environments.snake.snake import Snake

class SnakeGame:

    def __init__(self, W, H, training=True, fps=15):

        self.fps = fps # only if not training
        self.SCALE = 10
        self.training = training

        self.ROWS = H//self.SCALE
        self.COLS = W//self.SCALE

        self.W, self.H = self.COLS * self.SCALE, self.ROWS * self.SCALE

        self.COLOR_BLACK = (0,0,0)
        self.COLOR_WHITE = (255,255,255)
        self.COLOR_BLUE = (0,0,255)
        self.COLOR_RED = (255,0,0)

        self.GAME_TITLE = 'snake'
        self.world = pygame.display.set_mode((self.W, self.H))
        self.ACTIONS = 3
        pygame.display.set_caption(self.GAME_TITLE)
        self.action_space = {'n':3, 'ACTION_CODES':[0,1, 2], 'ACTIONS':['STRAIGHT', 'LEFT', 'RIGHT']}
        self.clock = pygame.time.Clock() # remove fps when training

        self.reset()

    # same API as OpenAI Gym
    # except step returns (state, reward, done, score)
    def step(self, action):
        pygame.event.get()
        if not self.training:
            self.clock.tick(self.fps)
        self.world.fill(self.COLOR_BLACK)
        is_dead, reward = self.snake.move(action, self.fruit)
        if(reward > 0):
            self.time_since_last_reward = 0
            self.score += reward
            self.fruit = self.generate_new_fruit(self.snake.indices)
        if is_dead or self.time_since_last_reward > self.ROWS*self.COLS:
            return self.get_screen(), -1, True, self.score
        self.time_since_last_reward += 1
        self.draw(self.snake.indices)
        self.draw_rect(self.fruit[0], self.fruit[1], self.COLOR_RED) # draw the fruit
        pygame.display.flip()
        return (self.get_screen(), reward, is_dead, self.score)

    def sample(self):
        return random.choice(range(3))

    def reset(self):
        self.world.fill(self.COLOR_BLACK)
        self.snake = Snake(self.COLS, self.ROWS)
        self.score = 0
        self.fruit = self.generate_new_fruit(self.snake.indices)
        self.time_since_last_reward = 0
        self.draw(self.snake.indices)
        self.draw_rect(self.fruit[0], self.fruit[1], self.COLOR_RED)
        #self.step(0)
        pygame.display.flip()
        return self.get_screen()


    def get_screen(self):
        return pygame.surfarray.array3d(pygame.display.get_surface())
    '''
    Converts the world to the pixel input fed to the neural network
    '''
    def cvtWorldToState(self):

        s = np.zeros((self.ROWS, self.COLS), dtype='uint32')
        for x, y in self.snake.indices:
            s[y,x] = 1
        s[self.snake.head[0], self.snake.head[1]] = 127
        s[self.fruit[1], self.fruit[0]] = 255
        return np.reshape(s, [10, 10, 1])

    def draw(self, indices):
        COLOR = self.COLOR_WHITE
        for pos in indices:
            self.draw_rect(pos[0], pos[1], COLOR)
            COLOR = tuple(int(COLOR[i]*0.95) for i in range(len(COLOR)))
        self.draw_head()

    # TODO: revise this
    def generate_new_fruit(self, indices):
        if len(indices) >= self.ROWS * self.COLS: return (-1,-1)

        new_fruit = (random.randint(0, self.COLS-1), random.randint(0, self.ROWS-1))
        while new_fruit in indices:
            new_fruit = (random.randint(0,self.COLS-1), random.randint(0,self.ROWS-1))
        return new_fruit

    def draw_head(self):
        if self.SCALE <= 1: return
        rect = pygame.Rect( self.snake.head[0] * self.SCALE + 1,
                            self.snake.head[1] * self.SCALE + 1,
                            max(0, self.SCALE-2),
                            max(0, self.SCALE-2))
        pygame.draw.rect(self.world, self.COLOR_BLUE, rect, 1)

    def draw_rect(self, x, y, color):
        SCALE = self.SCALE
        rect = pygame.Rect( SCALE*x,
                            SCALE*y,
                            SCALE,
                            SCALE)
        pygame.draw.rect(self.world, color, rect, 0)
