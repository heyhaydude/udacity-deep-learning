"""
Author: Uriel Sade
Date: July 2nd, 2017
"""
import random

class EpsilonGreedy:

    """
    Helper class to choose actions based on epsilon-greedy policy
    The default arg values are the same as DeepMind's original Atari DQN params
    Args:
        initial_value: Initial epsilon
        target_value: Final epsilon
        exploration_frames: Number of frames until epsilon reaches it's final value
    """
    def __init__(self, initial_value=1.0, target_value=0.1, exploration_frames=1e6, fixed=False, verbose=True):
        if initial_value < target_value and verbose:
            # raise ValueError("Initial value must be >= to target value")
            print("WARNING: \
            Initial epsilon ({}) is smaller than final epsilon ({})." \
            .format(initial_value, target_value))

        self.epsilon = initial_value
        self.explore = int(exploration_frames)
        self.delta = 0 if fixed else (target_value - initial_value)/exploration_frames
        self.target = target_value

    """
    Returns:
        True iff we should choose a random action based on e-greedy policy
    """
    def evaluate(self):
        self.epsilon = max(self.target, self.epsilon + self.delta)
        return random.random() <= self.epsilon

    def peek(self):
        return self.epsilon
