import torch
from game_v3 import Environment, Game
import numpy as np
import pygame
import sys
from network_model import PolicyNetwork

model = torch.load('./model_save/target_policy_net.pkl')
env = Game(Environment(window_size=(200, 200), obstructs=3, fps=60), True)
state = env.reset()
frames = 0
while True:
    for event in pygame.event.get():
        if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE) \
                or event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            env.reset()
            frames = 0
    action = model.get_action(state)
    print(action.shape, action)
    state, r, done, n = env.step(action)
    frames += 1
    if done:
        env.reset()
        frames = 0

    env.render()
