import random

NEGATIVE_REWARD = -1
POSITIVE_REWARD = +1

dirs = ['R', 'U', 'L', 'D']

class Snake:

    def __init__(self, W, H):
        self.W = W
        self.H = H
        start_x = self.W//2
        start_y = self.H//3
        self.direction = 'U'

        self.indices = []
        for _ in range(5):
            self.indices.append((start_x, start_y))
            start_y += 1
        self.head = self.indices[0]




    def update(self, new_head, fruit):
        self.indices.insert(0, new_head)
        self.head = new_head

        if new_head != fruit:
            self.indices.pop()
        is_dead = self.isDead()

        if is_dead:
            reward = NEGATIVE_REWARD
        elif new_head == fruit:
            reward = POSITIVE_REWARD
        else:
            reward = NEGATIVE_REWARD/100

        return (is_dead, reward)

    # Snake has died if it has run into itself
    # Out of bounds death is checked in the move functions
    # for simplicity
    def isDead(self):
        return len(self.indices) != len(set(self.indices))


    # Returns True if snake dies as a result of moving
    def move_left(self, fruit):
        self.direction = 'L'
        if self.head[0] == 0: return (True, NEGATIVE_REWARD)

        new_head = (self.head[0]-1, self.head[1])
        return self.update(new_head, fruit)

     # Returns True if snake dies as a result of moving
    def move_right(self, fruit):
        self.direction = 'R'
        if self.head[0] == self.W-1: return (True, NEGATIVE_REWARD)

        new_head = (self.head[0]+1, self.head[1])
        return self.update(new_head, fruit)

    # Returns True if snake dies as a result of moving
    def move_up(self, fruit):
        self.direction = 'U'
        if self.head[1] == 0: return (True, NEGATIVE_REWARD)

        new_head = (self.head[0], self.head[1]-1)
        return self.update(new_head, fruit)

    # Returns True if snake dies as a result of moving
    def move_down(self, fruit):
        self.direction = 'D'
        if self.head[1] == self.H-1: return (True, NEGATIVE_REWARD)

        new_head = (self.head[0], self.head[1]+1)
        return self.update(new_head, fruit)

    '''
    Snake has three moves:
    1) Move Left
    2) Move Right
    3) Move Straight

    In hindsight, this function could have been planned out better...
    '''
    def move(self, move, fruit):
        if move == 0:
            if self.direction == 'R':
                return self.move_right(fruit)
            elif self.direction == 'L':
                return self.move_left(fruit)
            elif self.direction == 'U':
                return self.move_up(fruit)
            elif self.direction == 'D':
                return self.move_down(fruit)
        elif move == 1:
            if self.direction == 'R':
                return self.move_up(fruit)
            elif self.direction == 'L':
                return self.move_down(fruit)
            elif self.direction == 'U':
                return self.move_left(fruit)
            elif self.direction == 'D':
                return self.move_right(fruit)
        elif move == 2:
            if self.direction == 'R':
                return self.move_down(fruit)
            elif self.direction == 'L':
                return self.move_up(fruit)
            elif self.direction == 'U':
                #print('here')
                return self.move_right(fruit)
            elif self.direction == 'D':
                return self.move_left(fruit)
        else: print("Invalid Move: ", move)
