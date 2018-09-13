import pygame
import numpy as np
from numpy.random import randint, ranf
from particle import Agent, Particle
import sys
import numba


class Game(object):
    def __init__(self, fps: int = 30, window_size: tuple = (600, 600)):
        self._FPS = fps
        self._WINDOW_SIZE = window_size
        pygame.init()
        pygame.key.set_repeat(10)
        self.frames_clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(self._WINDOW_SIZE)
        pygame.display.set_caption("Collide Lab (ง •_•)ง")
        self.obstructs = []
        self.agent = None
        self.target = None
        self._game_state = True
        self.font = pygame.font.SysFont('arial', 16)

    @numba.jit
    def add_character(self, character: Agent, character_type: str):
        if character_type == 'agent':
            self.agent = character
        elif character_type == 'obstruct':
            self.obstructs.append(character)
        elif character_type == 'target':
            self.target = character
        else:
            print("添加角色失败：", character)

    @numba.jit
    def draw_characters(self):
        """
        将agent和障碍物都进行绘制
        :return:
        """
        if self.agent is not None:
            self.agent.draw(self.screen)
        if self.target is not None:
            self.target.draw(self.screen)
        for this_obs in self.obstructs:
            this_obs.draw(self.screen)

    @numba.jit
    def display_info(self):
        """
        显示速度等信息
        """
        speed_info = "speed: {:.2f}, {:.2f}".format(self.agent.speed[0], self.agent.speed[1])
        text = self.font.render(speed_info, True, (0, 0, 0))
        speed_rect = pygame.Rect(10, 10, 100, 30)
        self.screen.blit(text, speed_rect)

        position_info = "position:{:.2f}, {:.2f}".format(self.agent.position[0], self.agent.position[1])
        text = self.font.render(position_info, True, (0, 0, 0))
        position_text = pygame.Rect(10, 30, 100, 60)  # 纵轴+30
        self.screen.blit(text, position_text)

    @numba.jit
    def keyboard_controller(self):
        new_speed = np.array([0, 0])
        if self._game_state:
            for event in pygame.event.get():
                if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE) \
                        or event.type == pygame.QUIT:  # esc 退出
                    print('game exit')
                    self._game_state = False
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        # new_speed = new_speed + np.array([-1, 0])
                        self.agent.step(np.array([-1, 0]))
                    elif event.key == pygame.K_d:
                        # new_speed = new_speed + np.array([1, 0])
                        self.agent.step(np.array([1, 0]))
                    elif event.key == pygame.K_w:
                        # new_speed = new_speed + np.array([0, -1])
                        self.agent.step(np.array([0, -1]))
                    elif event.key == pygame.K_s:
                        # new_speed = new_speed + np.array([0, 1])
                        self.agent.step(np.array([0, 1]))

                    elif event.key == pygame.K_q:  # 左上
                        # new_speed = new_speed + np.array([0, 1])
                        self.agent.step(np.array([-1, -1]))
                    elif event.key == pygame.K_e:  # 右上
                        # new_speed = new_speed + np.array([0, 1])
                        self.agent.step(np.array([1, -1]))
                    elif event.key == pygame.K_z:  # 左下
                        # new_speed = new_speed + np.array([0, 1])
                        self.agent.step(np.array([-1, 1]))
                    elif event.key == pygame.K_c:  # 右下
                        # new_speed = new_speed + np.array([0, 1])
                        self.agent.step(np.array([1, 1]))



                        # # for i in range(2):
                        # #     # 限制新速度绝对值最大为3
                        # #     new_speed[i] = int(new_speed[i]) / abs(new_speed[i]) * 10 if abs(new_speed[i]) > 10 else new_speed[i]
                        # return new_speed

    def check_limit(self):
        for this_obs in self.obstructs:
            this_obs.check_bound(self._WINDOW_SIZE)

    def check_runtime_state(self):
        if self.agent is None:
            raise Exception("游戏没有添加agent")

    @numba.jit
    def play(self):
        if self._game_state:
            # new_speed = self.keyboard_controller()
            self.keyboard_controller()
            # self.agent.step(new_speed)

            for i, this_obs in enumerate(self.obstructs):
                this_obs.step()
                #  检查碰撞摧毁红色
                if np.linalg.norm(self.agent.position - this_obs.position) <= 31:
                    del self.obstructs[i]

            # 达到目标点
            if np.linalg.norm(self.agent.position - self.target.position) <= 31:
                del self.target
                # 重新生成新目标
                init_p = randint(35, 365), randint(35, 365)  # 防止在边界生成
                self.add_character(Agent(init_position=init_p,
                                         color='g', name='target'), 'target')

            self.screen.fill((255, 255, 255))
            self.draw_characters()
            self.display_info()

            pygame.display.update()
            self.frames_clock.tick(self._FPS)


def game_test():
    game = Game(fps=60)
    game.add_character(Agent(init_position=np.array([300, 500]), color='b', name='agent'), 'agent')
    game.add_character(Agent(init_position=np.array([300, 100]), color='g', name='target'), 'target')
    for i in range(10):
        init_v = randint(-3, 3, 2)
        # init_p = randint(35, 365), randint(35, 365)  # 防止在边界生成
        init_p = randint(35, 365, 2)
        game.add_character(Agent(init_speed=init_v, init_position=init_p,
                                 color='r', decay_speed=False,
                                 name=str(i)),
                           character_type='obstruct')
    while True:
        game.play()

if __name__ == '__main__':
    game_test()

