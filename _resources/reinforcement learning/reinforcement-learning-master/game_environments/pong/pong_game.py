import pygame
import random

#TODO: map ball's y_speed to impact position along the paddles

def linear_map(val, x1, x2, y1, y2):
    slope = (y1-y2)/(x1-x2)
    y = slope*(val - x1) + y1
    return int(y)

class Pong:

    """
    Args:
        W: The width  of the game's window
        H: The height of the game's window
    """
    def __init__(self, W, H, training=True, fps=15):
        self.WINDOW_WIDTH  = W
        self.WINDOW_HEIGHT = H
        self.PADDLE_WIDTH  = W//35
        self.PADDLE_HEIGHT = H//7

        # TODO: Change to radius and make the ball a circle
        self.BALL_WIDTH = W//60 + H//60

        self.PADDLE_SENSITIVITY = max(1, self.PADDLE_HEIGHT//4)
        self.BALL_SPEED_X = max(1, self.BALL_WIDTH)
        self.BALL_SPEED_Y = max(1, self.BALL_WIDTH)

        self.COLOR_BLACK = (0,0,0)
        self.COLOR_WHITE = (255,255,255)
        self.COLOR_BLUE = (0,0,255)
        self.COLOR_RED = (255,0,0)

        self.GAME_TITLE = 'pong'
        self.world = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        self.action_space = {'n':2, 'ACTION_CODES':[0,1], 'ACTIONS':['DOWN', 'UP']}

        pygame.display.set_caption(self.GAME_TITLE)

        self.training = training
        self.fps = fps
        self.clock = pygame.time.Clock()

        self.reset()

    """
    Performs an action in the enviromnent causing a transition to the next state
    Returns:
        (s, r, t, sc):
            s -> new state (ndarray)
            r -> Reward(state, action)
            t -> is this a terminal state, i.e. won or lost (boolean)
            sc-> the running score (# times successfully blocked)
    """
    def step(self, action):
        pygame.event.get() # flushes the event queue
        if not self.training:
            self.clock.tick(fps)
        self.world.fill(self.COLOR_BLACK)
        self.move_player_paddle(action)
        self.move_opponent_paddle()
        self.move_ball()
        reward, terminal = self.check_status()
        self.draw()
        return (self.get_state(), reward, terminal, self.score)

    """
    Resets the game environment
    Returns:
        The initial state (ndarray)
    """
    def reset(self):
        self.player_paddle = {
                               'x':self.WINDOW_WIDTH//10,
                               'y':self.WINDOW_HEIGHT//2 - self.PADDLE_HEIGHT//2
                            }
        self.opponent_paddle = {
                               'x':self.WINDOW_WIDTH - self.player_paddle['x'],
                               'y':self.player_paddle['y']
                            }
        self.ball = {
                      'x': self.opponent_paddle['x'] - self.BALL_WIDTH//2,
                      'y': self.WINDOW_HEIGHT//2 - self.BALL_WIDTH//2,
                      'x_direction': -1,
                      'y_direction': random.sample([-1,1], 1)[0]
                     }
        self.score = 0
        self.draw()
        return self.get_state() # game's inital state

    """
    Returns:
        A random action
    """
    def sample(self):
        return random.choice(range(self.action_space['n']))
    """
    Moves the player's paddle either up or down according to the action
    Args:
        action: the action being performed
    Raises:
        ValueError: if the action is invalid
    """
    def move_player_paddle(self, action):
        if action == 0 and self.player_paddle['y'] + self.PADDLE_HEIGHT < self.WINDOW_HEIGHT:
            self.player_paddle['y'] += self.PADDLE_SENSITIVITY
        elif action == 1 and self.player_paddle['y'] > 0:
            self.player_paddle['y'] -= self.PADDLE_SENSITIVITY

        if action not in self.action_space['ACTION_CODES']:
            raise ValueError('Error: Invalid action: {}'.format(action))
        return

    """
    Moves the opponent's paddle in the diraction of the ball
    """
    def move_opponent_paddle(self):
        distance =  self.ball['y'] - (self.opponent_paddle['y'] + self.PADDLE_HEIGHT//2)

        if distance > 0 and self.opponent_paddle['y'] + self.PADDLE_HEIGHT < self.WINDOW_HEIGHT:
            self.opponent_paddle['y'] += min(self.PADDLE_SENSITIVITY, abs(distance))
        elif self.opponent_paddle['y'] > 0:
            self.opponent_paddle['y'] -= min(self.PADDLE_SENSITIVITY, abs(distance))
        return

    """
    Moves the ball according to its direction vector
    """
    def move_ball(self):

        self.ball['y'] += self.ball['y_direction']*self.BALL_SPEED_Y
        self.ball['x'] += self.ball['x_direction']*self.BALL_SPEED_X

        if self.ball['y'] < 0:
            self.ball['y'] = 0
            self.ball['y_direction'] = 1
        elif self.ball['y'] + self.BALL_WIDTH > self.WINDOW_HEIGHT:
            self.ball['y'] = self.WINDOW_HEIGHT - self.BALL_WIDTH
            self.ball['y_direction'] = -1

        return
    """
    Checks the game's status and updates the ball's direction if necessary
    Returns:
        (reward, terminal):
            reward   -> the reward from the last action
            terminal -> is this a terminal state, i.e. won or lost (boolean)
    """
    def check_status(self):
        reward = 0
        ball_midpoint_x = self.ball['x'] + self.BALL_WIDTH//2
        player_paddle_right = self.player_paddle['x'] + self.PADDLE_WIDTH
        opponent_paddle_left = self.opponent_paddle['x']

        if self.ball['x'] <= 0:
            return (-1, True)
        elif self.ball['x'] <= player_paddle_right \
            and self.intersects_ball(self.player_paddle):
            reward = 1 if self.ball['x_direction'] == -1 else 0
            self.ball['x_direction'] = 1
            self.score += 1

        if ball_midpoint_x > opponent_paddle_left:
            return (1000, True)
        elif self.ball['x'] + self.BALL_WIDTH >= opponent_paddle_left \
            and self.intersects_ball(self.opponent_paddle):
            reward = -1 if self.ball['x_direction'] == 1 else 0
            self.ball['x_direction'] = -1

        return (reward, False)

    """
    Args:
        paddle: The paddle in question
    Returns:
        True iff the paddle intersects the ball and adjusts the ball's y-speed
        if they intersect.
    """
    def intersects_ball(self, paddle):
        ball_l = self.ball['x']
        ball_r = ball_l + self.BALL_WIDTH
        ball_t = self.ball['y']
        ball_b = ball_t + self.BALL_WIDTH
        paddle_l = paddle['x']
        paddle_r = paddle_l + self.PADDLE_WIDTH
        paddle_t = paddle['y']
        paddle_b = paddle_t + self.PADDLE_HEIGHT

        intersects = ball_l <= paddle_r and ball_r > paddle_l \
            and ball_b > paddle_t and ball_t < paddle_b

        if intersects:
            dist_from_paddle_center = abs((ball_t + ball_b // 2) - (paddle_t + paddle_b // 2))
            self.BALL_SPEED_Y = linear_map(dist_from_paddle_center, 0, self.PADDLE_HEIGHT//2, 0, self.BALL_WIDTH * 2)

        return intersects

    """
    Returns:
        The raw pixel data of the game (ndarray)
    """
    def get_state(self):
        return pygame.surfarray.array3d(pygame.display.get_surface())

    """
    Draws the paddles and ball
    """
    def draw(self):
        self.draw_rect( self.player_paddle['x'], self.player_paddle['y'],
                        self.PADDLE_WIDTH, self.PADDLE_HEIGHT, self.COLOR_WHITE
                        )
        self.draw_rect( self.opponent_paddle['x'], self.opponent_paddle['y'],
                        self.PADDLE_WIDTH, self.PADDLE_HEIGHT, self.COLOR_RED
                        )
        self.draw_rect( self.ball['x'], self.ball['y'],
                        self.BALL_WIDTH, self.BALL_WIDTH, self.COLOR_BLUE
                        )
        pygame.display.flip()
        return

    """
    Helper function to draw rectangles on the screen
    Args:
        x: The upper side of the rectangle
        y: The left  side of the rectangle
        W: The width of the rectangle (extends to the right of x)
        H: The height of the rectangle (extends to the left of y)
        color: an RGB tuple representing the rectangle's color
    """
    def draw_rect(self, x, y, W, H, color):
        rect = pygame.Rect( x, y, W, H )
        pygame.draw.rect(self.world, color, rect, 0)
        return
