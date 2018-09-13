from particle import Particle
import pygame
import numpy as np
import sys
from pygame import locals
import numba
from random import random
from numpy.random import randint


class Game(object):
    def __init__(self, window_size, background_color, fps):
        self.__BACKGROUND_COLOR = background_color
        self.__FPS = fps
        self.__WINDOW_SIZE = window_size
        self.frames_clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(self.__WINDOW_SIZE)
        pygame.display.set_caption("Collide Lab (ง •_•)ง")
        self.__Particles = []
        self.particles = []  # rect obj
        self.__game_state = True

    def create_player(self, particle: Particle):
        self.__Particles.append(particle)

    def display_players(self):
        for this_particle in self.__Particles:
            p, v = this_particle.get_state()
            pygame.draw.circle(self.screen, this_particle.color, p, this_particle.r)

    def update_game(self, velocity_list=None, relative: bool = False):
        for event in pygame.event.get():
            if ((event.type == locals.KEYDOWN) and (event.key == locals.K_ESCAPE)) or (event.type == pygame.QUIT):
                sys.exit()

        for i, this_particle in enumerate(self.__Particles):
            if velocity_list is not None:
                if relative:
                    this_particle.step(velocity_list[i])
                else:
                    new_v = [this_particle.cur_speed[0] + velocity_list[i][0],
                             this_particle.cur_speed[1] + velocity_list[i][1]]
                    this_particle.step(new_v)
            else:
                this_particle.step(this_particle.cur_speed)
            this_particle.check_bound(self.__WINDOW_SIZE)

        self.screen.fill(self.__BACKGROUND_COLOR)
        self.display_players()
        pygame.display.update()
        self.frames_clock.tick(self.__FPS)

    def play_game(self):
        """
        修复
        """
        self.display_players()
        # self.screen.fill(self.__BACKGROUND_COLOR)
        while True:
            for event in pygame.event.get():
                if ((event.type == locals.KEYDOWN) and (event.key == locals.K_ESCAPE)) \
                        or (event.type == pygame.QUIT):
                    sys.exit()
                    # self.update_game()
                    # self.frames_clock.tick(self.__FPS)


if __name__ == '__main__':
    game = Game((600, 400), (0, 0, 0), 144)
    game.create_player(Particle(25, [1, 1], [300, 200]))
    game.create_player(Particle(10, [2, 2], [100, 200]))
    for i in range(100):
        r = np.random.randint(5, 10)
        init_v = randint(-5, 5, 2)
        init_p = [randint(20, 580), randint(20, 380)]  # 避免在边界生成
        game.create_player(Particle(r, init_v, init_p))
    while True:
        game.update_game()
