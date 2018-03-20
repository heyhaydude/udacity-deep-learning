# Reinforcement Learning with TensorFlow #

An attempt at implementing a DQN similar to DeepMind's DQN algorithm from the papers ["Playing Atari with Deep Reinforcement Learning"](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) and ["Human-level control through deep reinforcement learning"](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) using TensorFlow.

Special thanks to David Silver from DeepMind for [his course](https://youtu.be/2pWv7GOvuf0?list=PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-) on Reinforcement Learning.

This is continuing work in progress.

`R(s, a)` =  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;+1 if the agent has won or achieved some predefined goal as a result of the action `a`.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;in state `s` (e.g. snake game -> snake just   ate some food)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-1 if agent has lost as a result of the action `a` in state `s`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(e.g. snake game -> snake went out of bounds or ran into itself)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-0.01 otherwise

Run with the `--help` flag to see possible command line args.

If you want to download a saved checkpoint, the following [dropbox folder](https://www.dropbox.com/sh/olqliwn0m8pyy06/AADw9FqLl2Sjrr3JmiMj5rdRa?dl=0) contains the checkpoint weights from 10 hours of training. Move the uncompressed folder named `100_100` inside the `./saved_checkpoints/snake/` directory. It's ~ 500 MB uncompressed.

### Video: After 24 hours of training (on a cpu... ):

[![After 24 hours](https://i.ytimg.com/vi/IDrds7-HSTg/hqdefault.jpg?sqp=-oaymwEXCNACELwBSFryq4qpAwkIARUAAIhCGAE=&rs=AOn4CLA4-usZUfxm-G5CeIJsO3xwuvBHZw)](https://youtu.be/IDrds7-HSTg)

#### Score distribution after 24 hours of training:

![Score dist. after 24 hours of training](./statistics/24_hours.png)

#### Score distribution after 10 hours of training:

![Score dist. after 10 hours of training](./statistics/10_hours.png)

#### Score distribution after 7 hours of training:

![Score dist. after 7 hours of training](./statistics/7_hours.png)

#### Initial Score Distribution (random actions):

![Initial Score Distribution (random actions)](./statistics/random_actions.png)

More soon.
