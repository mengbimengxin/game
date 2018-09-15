from random import randint
import pygame
import numba
import numpy as np


class Particle(object):
    def __init__(self, r, init_speed=(0, 0), init_position=(0, 0)):
        self.init_speed = init_speed
        self.cur_speed = init_speed
        self.init_position = init_position
        self.cur_position = init_position
        self.r = r
        self.speeds_history = []
        self.positons_history = []
        self.store_state()
        self.color = (randint(0, 255), randint(0, 255), randint(0, 255))

    def step(self, velocity):
        self.cur_speed = velocity
        # self.cur_position += velocity
        self.cur_position[0] = velocity[0] + self.cur_position[0]
        self.cur_position[1] = velocity[1] + self.cur_position[1]
        self.store_state()

    def store_state(self):
        self.speeds_history.append(self.cur_speed)
        self.positons_history.append(self.cur_position)

    def get_state(self):
        """
        获得该粒子最近的状态
        :return: 最近位置(二元元祖),最近速度(二元元祖)
        """
        return self.cur_position, self.cur_speed

    def check_bound(self, window_size):
        for i in range(2):
            if self.cur_position[i] - self.r < i or self.cur_position[i] + self.r > window_size[i]:
                self.cur_speed[i] = -self.cur_speed[i]


class Agent(object):
    def __init__(self, d=35,
                 init_position: np.ndarray = np.array([0, 0]),
                 init_speed: np.ndarray = np.array([0, 0]),
                 color: str = 'b',
                 decay_speed: bool = True,
                 activity_rect=(600, 600),
                 name: str = ''):
        self.position = init_position
        self.speed = init_speed
        self.d = d
        self._DECAY = decay_speed
        self._activity_rect = activity_rect
        # self.font = pygame.font.SysFont('arial', 13)
        self.name = name

        if color == 'b':
            self._ball = pygame.image.load("./particle_img/blue.png")
        elif color == 'r':
            self._ball = pygame.image.load("./particle_img/red.png")
        elif color == 'g':
            self._ball = pygame.image.load("./particle_img/green.png")

        # self.rect = self.__ball.get_rect().move_ip(self.position[0], self.position[1])  # 移动到初试位置
        self.rect = self._ball.get_rect()
        self.rect.move_ip(self.position[0], self.position[1])

    def step(self, speed: np.ndarray = None):
        if speed is not None:
            self.speed = 0.5 * self.speed + 0.5 * speed
        if self._DECAY:
            for i in range(2):
                self.speed[i] = 0 if abs(self.speed[i]) < 1e-2 else self.speed[i] * 0.9
        self.position = self.position + self.speed
        self.check_bound(self._activity_rect)

        self.rect.center = self.position

    def check_bound(self, window_size):
        for i in range(2):
            if self.position[i] - 0.5 * self.d < 0 or self.position[i] + 0.5 * self.d > window_size[i]:
                # 调整速度方向
                self.speed[i] = -self.speed[i]
                # 防止位置出界
                self.position[i] = window_size[i] if self.position[i] > window_size[i] else self.position[i]
                self.position[i] = 0 if self.position[i] < 0 else self.position[i]

    def draw(self, screen: pygame.SurfaceType, name: bool = False):
        screen.blit(self._ball, self.rect)
        if name:
            text = self.font.render(self.name, True, (0, 0, 0))
            text_rect = self.rect
            text_rect.topleft = self.rect.center
            screen.blit(text, self.rect)


class Radar(object):
    # 设置雷达坐标和半径
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r
        self.xx = []
        self.yy = []
        self.x_x = []
        self.y_y = []
        # self.env = env
        # pygame.key.set_repeat(500, 10)

    # 绘制雷达,item是指screen
    def draw(self, item, color=(255, 0, 0)):
        for k in np.arange(0, 360, 10):
            pos_xx = self.r * np.cos(k * np.pi / 180)
            pos_yy = self.r * np.sin(k * np.pi / 180)
            # self.dx.append(pos_xx)
            # self.dy.append(pos_yy)
            pos_x = pos_xx + self.x
            pos_y = pos_yy + self.y
            pygame.draw.line(item, color, [self.x, self.y], [pos_x, pos_y])

    # 键盘响应控制移动
    # def control(self):
    #     for event in pygame.event.get():
    #         if event.type == pygame.KEYDOWN:
    #             if event.key == pygame.K_LEFT:
    #                 self.x -= 10
    #             elif event.key == pygame.K_RIGHT:
    #                 self.x += 10
    #             elif event.key == pygame.K_UP:
    #                 self.y -= 10
    #
    #             elif event.key == pygame.K_DOWN:
    #                 # self.y += 10/

    # 计算障碍物与雷达的交点，参数Item为障碍物，封装时方法中Item.--的参数名要改过来
    def calculate(self, Item):
        # dis_radar = [self.r]*36
        for p in np.arange(0, 360, 10):
            pos_xx = self.r * np.cos(p * np.pi / 180)
            pos_yy = self.r * np.sin(p * np.pi / 180)
            self.xx.append(pos_xx)
            self.yy.append(pos_yy)
            # pos_x = pos_xx + self.x
            # pos_y = pos_yy + self.y
        dd = np.sqrt(np.square(self.x - Item.position[0]) + np.square(self.y - Item.position[1]))
        if dd <= self.r + (Item.d / 2):
            for q in np.arange(0, 36):
                A = np.square(self.xx[q]) + np.square(self.yy[q])
                B = 2 * ((self.xx[q]) * (-(Item.position[0] - self.x)) + (self.yy[q]) * (-(Item.position[1] - self.y)))
                C = (Item.position[0] - self.x) ** 2 + (Item.position[1] - self.y) ** 2 - (Item.d / 2) ** 2
                the = np.square(B) - 4 * A * C
                # u1 = (-B - np.sqrt(the) / (2 * A))
                # u2 = (-B + np.sqrt(the) / (2 * A))
                if the == 0:
                    u = -B / (2 * A)
                    if u >= 0 and u <= 1:
                        xx = self.xx[q] * u
                        yy = self.yy[q] * u
                        self.x_x.append(xx)
                        self.y_y.append(yy)
                        print(xx, yy)
                        # print(u)
                elif the > 0:
                    u1 = ((-B) - np.sqrt(the)) / (2 * A)
                    u2 = ((-B) + np.sqrt(the)) / (2 * A)
                    # print("u:",u1,u2)
                    # print(the)
                    if (u1 >= 0 and u1 <= 1) and (u2 >= 0 and u2 <= 1):
                        xx = self.xx[q] * u1
                        yy = self.yy[q] * u1
                        self.x_x.append(xx)
                        self.y_y.append(yy)
                        print(xx, yy)

                    elif (u1 >= 0 and u1 <= 1) and (u2 > 1 or u2 < 0):
                        xx = self.xx[q] * u1
                        yy = self.yy[q] * u1
                        self.x_x.append(xx)
                        self.y_y.append(yy)
                        print(xx, yy)
                    elif (u2 >= 0 and u2 <= 1) and (u1 > 1 or u1 < 0):
                        xx = self.xx[q] * u2
                        yy = self.yy[q] * u2
                        self.x_x.append(xx)
                        self.y_y.append(yy)
                        print(xx, yy)
                else:
                    self.x_x.append()

        return self.x_x, self.y_y
