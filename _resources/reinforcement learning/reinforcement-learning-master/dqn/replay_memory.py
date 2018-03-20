"""
Author: Uriel Sade
July 1st, 2017
"""
import numpy as np
import random

class ReplayMemory:
    """
    Replay Memory Store

    Stores (s_t1, a_t1, r_t1, s_t2) tuples
        s_t1 -> State at time t1
        a_t1 -> Action taken at time t1
        r_t1 -> Reward received at time t1
        s_t2 -> The resulting state at time (t1 + 1) (i.e. t2)
    Args:
        capacity: The maximum entry capacity of the replay memory
        batch_size: The desired size of each sampled minibatch
    """
    def __init__(self, capacity=50000):

        self.MAX_CAPACITY = max(capacity, 1000)
        self.internal_memory = []
        self.replace_index = 0

    """ Adds an entry to the replay memory
    Args:
        addition: A tuple containing (s_t1, a_t1, r_t1, s_t2)
    """
    def add(self, addition):
        if len(self.internal_memory) < self.MAX_CAPACITY:
            self.internal_memory.append(addition)
        else:
            self.internal_memory[self.replace_index] = addition
            self.replace_index = (self.replace_index + 1) % self.MAX_CAPACITY

    """ Generated a minibatch of past <s, a, r, s2> memories
    Args:
        batch_size (optional): the desired size of the minibatch
    Returns:
        A list from the replay memory containing
          random (s_t1, a_t1, r_t1, s_t2) tuples
    """
    def get_minibatch(self, batch_size=32):
        if len(self.internal_memory) < batch_size:
            batch_size = len(self.internal_memory)
        indices = np.random.choice( len(self.internal_memory),
                                    batch_size, replace=False )

        return [self.internal_memory[i] for i in indices]
