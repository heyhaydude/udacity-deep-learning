"""
Author: Uriel Sade
Date: July 4rd, 2017
"""

#TODO: move this into a class and run it's methods from a main script

from dqn.neural_net import NeuralNet
from dqn.replay_memory import ReplayMemory
from dqn.epsilon_greedy import EpsilonGreedy
from game_environments.pong.pong_game import Pong
from game_environments.snake.snake_game import SnakeGame
import util.parser as parser
import util.stats_saver as stats_saver

import numpy as np
import cv2
import time

def main():

    REPLAY_CAPACITY = 100000
    INITIAL_EPSILON = 1.0
    TARGET_EPSILON  = 0.1
    EXPLORATION_FRAMES = 1e6
    BATCH_SIZE = 32
    GAMMA = 0.97
    LR = 0.0005

    training, game, verbose, fps, W, H = parser.get_arguments()
    training = parser.str2bool(training)
    start_time = time.time()

    max_score = 0
    games_played = 0
    frame_iterations = 0
    scores = {}

    print("Training: ", training)

    if game == 'pong':
        env = Pong(W, H)
    elif game == 'snake':
        env = SnakeGame(W,H, training=training, fps=fps)
    else:
        print('Invalid game title')
        return

    nn = NeuralNet(W,H, env.action_space['n'], env.GAME_TITLE, n_channels=1,
                                                               gamma=GAMMA,
                                                               learning_rate=LR,
                                                               verbose=verbose)

    replay_memory = ReplayMemory(capacity=REPLAY_CAPACITY)
    epsilon_greedy = EpsilonGreedy( initial_value=INITIAL_EPSILON,
                                    target_value=TARGET_EPSILON,
                                    exploration_frames=EXPLORATION_FRAMES)
    try:
        s = env.reset()
        s = process(s, W, H)
        while True:
            # make 10 moves, then train on a minibatch
            for i in range(10):
                take_random = epsilon_greedy.evaluate()
                if training and take_random:
                    a = env.sample()
                else:
                    a = nn.predict([s])[0]
                s1, r, t, score = env.step(a)
                s1 = process(s1, W, H)
                replay_memory.add((s, a, r, s1, t))
                frame_iterations+=1
                if not t:
                    s = s1
                else:
                    max_score = max(max_score, score)
                    games_played += 1
                    scores[score] = scores.get(score, 0) + 1
                    e_value = 0 if not training else epsilon_greedy.peek()
                    print("\rMax Score: {:3} || Last Score: {:3} || Games Played: {:7} Iterations: {:10} Epsilon: {:.5f} Scores: {}" \
                        .format(max_score, score, games_played, frame_iterations, e_value, str(scores)),
                        end="\n" if verbose or games_played % 1000 == 0 else "")
                    s = env.reset()
                    s = process(s, W, H)
            if training and frame_iterations > REPLAY_CAPACITY // 2:
                batch = replay_memory.get_minibatch(batch_size=BATCH_SIZE)
                loss = nn.optimize(batch)

    except KeyboardInterrupt:
        if training:
            nn.save()
            print("\nCheckpoint saved")
        nn.close_session()
        stats_saver.save_to_file(env.GAME_TITLE, max_score, games_played, frame_iterations, scores, training, start_time)
        print("Session closed")

"""
Note: If for some reason OpenCV is not used, comment out the first 3 lines
      in this function (as well as the import), reshape the state into
      [W, H, 3] and pass in `3` as `n_channels` to the neural net model.
      Doing so will take longer to train.
"""
def process(state, W, H):
    state = cv2.resize(state, (W, H))
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    #cv2.imwrite('test.png', state)
    #state = cv2.normalize(state, state, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    state = np.reshape(state, [W, H, 1])
    return state


if __name__ == "__main__":
    main()
