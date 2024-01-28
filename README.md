# Tetris AI

## Introduction

Tetris AI repo is a project that aims to create a Tetris AI that can play the
game of Tetris, directly on "TETR.IO" (https://tetr.io/).

## To do list

- [x] Create a Tetris game engine
- [ ] Create a neural network that play the game
  - [X] Create the Torch Env wrapper for the Tetris game engine
  - [X] Create the Actor Critic Neural Network
  - [X] Create the Replay Buffer
  - [X] Create the PPO Trainer
  - [x] Combine the Tetris game engine and the NN Agent
  - [ ] Add time dimension for the NN Agent
  - [ ] Add more than a single batch for training
  - [ ] Add a way to save the NN Agent
  - [ ] Add a way to load the NN Agent
  - [ ] Add a way to visualize the trained NN Agent
  - [ ] Upgrade the Replay Buffer to recall past events as well
  - [ ] Modify the PPO Algorithm to accept sparse events, rather than continous
- [ ] Create a vision processing component to read the game state from the
  screen
- [ ] Combine the neural network and the vision processing component to play 
  directly on the website.