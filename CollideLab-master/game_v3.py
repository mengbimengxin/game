import pygame
import numpy as np
from numpy.random import randint, ranf
from particle import Agent, Particle, Radar
import sys
import numba


class Box:
    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape


class Environment(object):
    def __init__(self,
                 window_size: np.ndarray = np.array([500, 500]),
                 fps: int = 60,
                 obstructs: int = 10,
                 display: bool = True):
        self._FPS = fps
        self._WINDOW_SIZE = window_size
        self._GAME_STATE = True
        self._PLAYING = True
        self._obs_num = obstructs

        self.action_space = Box(-1, 1, (2, 1))
        self.observation_space = Box(-1, 1, (2 * obstructs + 2, 1))

        self._obstructs = []
        self._agent = None
        self._radar = None
        self._target = None
        # self.font = pygame.font.SysFont('arial', 16)
        self._used_time = 0  # 使用fps总数总数
        self._state = None
        self.reset()

    @property
    def fps(self):
        return self._FPS

    @property
    def used_time(self):
        return self._used_time

    @property
    def window_size(self):
        return tuple(self._WINDOW_SIZE)

    @property
    def obstructs(self):
        return self._obstructs

    @property
    def agent(self):
        return self._agent

    @property
    def target(self):
        return self._target

    @property
    def radar(self):
        return self._radar

    def reset(self):
        """
        重置游戏
        """

        # 添加agent
        del self._agent
        self._agent = None
        self._agent = Agent(init_position=np.array(self._WINDOW_SIZE) / 2,
                            init_speed=np.array([0, 0]),
                            color='b',
                            decay_speed=True,
                            activity_rect=self._WINDOW_SIZE)

        #添加雷达
        del self._radar
        self._radar = None
        self._radar = Radar(self._agent.position[0],self._agent.position[1],100)


        # 设置目标
        del self._target
        self._target = None
        init_p = np.array([randint(35, self._WINDOW_SIZE[0] - 35), randint(35, self._WINDOW_SIZE[1] - 35)])
        self._target = Agent(init_position=init_p,
                             init_speed=np.array([0, 0]),
                             color='g',
                             decay_speed=True,
                             activity_rect=self._WINDOW_SIZE)
        # 添加障碍物
        del self._obstructs
        self._obstructs = []
        for i in range(self._obs_num):
            init_v = randint(-2, 2, 2) * 0.5
            # init_v = np.array([0,0])
            init_p = np.array([randint(35, self._WINDOW_SIZE[0] - 35), randint(35, self._WINDOW_SIZE[1] - 35)])
            while np.linalg.norm(init_p - self._agent.position) < 1.2 * self._agent.d:
                init_p = np.array([randint(35, self._WINDOW_SIZE[0] - 35), randint(35, self._WINDOW_SIZE[1] - 35)])
            self._obstructs.append(Agent(init_speed=init_v,
                                         init_position=init_p,
                                         color='r',
                                         decay_speed=False,
                                         name=str(i),
                                         activity_rect=self._WINDOW_SIZE))

    def check_limit(self):
        for this_obs in self._obstructs:
            this_obs.check_bound(self._WINDOW_SIZE)

    @numba.jit
    def check_collide(self):
        # 检测障碍物碰撞，并且重新刷新到新的位置
        n = 0
        arrived = False
        for i, this_obs in enumerate(self._obstructs):
            if np.linalg.norm(self._agent.position - this_obs.position) <= 0.5 * (self._agent.d + this_obs.d):
                n += 1
                init_p = np.array([randint(35, self._WINDOW_SIZE[0] - 35), randint(35, self._WINDOW_SIZE[1] - 35)])
                this_obs.position = init_p

        if np.linalg.norm(self._agent.position - self._target.position) <= 0.65 * (self._agent.d + self._target.d):
            # init_p = np.array([randint(35, self._WINDOW_SIZE[0] - 35), randint(35, self._WINDOW_SIZE[1] - 35)])
            # self._target.position = init_p
            self.reset()
            arrived = True
        return n, arrived

    def reset_target(self):
        init_p = np.array(randint(35, self._WINDOW_SIZE[0] - 35), randint(35, self._WINDOW_SIZE[1] - 35))
        self._target.position = init_p

    def get_state(self):
        self._state = []
        for this_obs in self.obstructs:
            self._state.append(this_obs.position)
        self._state = ((np.array(self._state) - self._agent.position) / self._WINDOW_SIZE).flatten()
        relative_agent_pos = (self._target.position - self._agent.position) / self._WINDOW_SIZE
        self._state = np.concatenate((self._state, relative_agent_pos))
        return self._state

    @numba.jit
    def step(self, action):
        # self._state = []
        self._agent.step(action)
        self._agent.check_bound(self._WINDOW_SIZE)

        for i, this_obs in enumerate(self._obstructs):
            this_obs.step()
            this_obs.check_bound(self._WINDOW_SIZE)
            # self._state.append(this_obs.position)
            self._radar.calculate(this_obs)     #各个障碍物与雷达交点坐标
        self._radar.calculate(self._target)     #目标与雷达交点坐标
        self.get_state()
        n, arrived = self.check_collide()
        self.check_limit()

        self._used_time += 1
        reward = self.compute_reward(n, arrived)
        return self._state, reward, arrived, n

    def compute_reward(self, n, arrived):
        """
        计算奖励
        :param arrived: 是否到达目标
        :param n: 这个时刻的碰撞次数
        :return: 奖励值
        """
        # r = [0] * 4
        d_target = np.linalg.norm(self._state[-2::])
        # r[0] -= np.log(d_target + 1.5)  # 距离惩罚
        # r[1] -= n * 10  # 碰撞惩罚
        # r[2] -= 1  # 时间惩罚
        if d_target <= 0.15:
            reward = np.cos(d_target / 0.4 * np.pi)
        else:
            reward = -np.log(d_target + 1.3) + 0.5 * np.log(1 + np.sqrt(2))
        if arrived:
            reward = 1
        if n > 0:
            reward = -1
        # reward = np.array([reward])
        return reward


class Game(object):
    def __init__(self, env: Environment, play=False):
        if play:
            pygame.init()
            pygame.key.set_repeat(20)
            self.screen = pygame.display.set_mode(env.window_size)
            pygame.display.set_caption("Collide Lab")
            self.font = pygame.font.SysFont('arial', 16)
            self.frames_clock = pygame.time.Clock()
        self.env = env
        self.RUNNING = True

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def _draw_characters(self):
        """
        绘制agent，target以及障碍物还有雷达
        :return:
        """
        if self.env.radar is not None:
            self.env.radar.draw(self.screen)
        if self.env.agent is not None:
            self.env.agent.draw(self.screen)
        if self.env.target is not None:
            self.env.target.draw(self.screen)
        for this_obs in self.env.obstructs:
            this_obs.draw(self.screen)

    @numba.jit
    def display_info(self):
        """
        显示速度等信息
        """
        speed_info = "speed: {:.2f}, {:.2f}".format(self.env.agent.speed[0], self.env.agent.speed[1])
        text = self.font.render(speed_info, True, (0, 0, 0))
        speed_rect = pygame.Rect(10, 10, 100, 30)
        self.screen.blit(text, speed_rect)

        position_info = "position:{:.2f}, {:.2f}".format(self.env.agent.position[0], self.env.agent.position[1])
        text = self.font.render(position_info, True, (0, 0, 0))
        position_text = pygame.Rect(10, 30, 100, 60)  # 纵轴+30
        self.screen.blit(text, position_text)


    @numba.jit
    def play(self):
        if self.RUNNING:
            info = self.env.step(randint(-2, 2, 2))
            print(info)

        self.screen.fill((255, 255, 255))
        self._draw_characters()
        # self.display_info()
        pygame.display.update()
        self.frames_clock.tick(self.env.fps)

    def step(self, actions: np.ndarray):
        return self.env.step(actions)

    def reset(self):
        self.env.reset()
        return self.env.get_state()

    def render(self):
        """
        显示游戏
        """
        self.screen.fill((255, 255, 255))
        self._draw_characters()
        self.display_info()
        pygame.display.update()
        self.frames_clock.tick(self.env.fps)


def game_test():
    env = Environment(window_size=(800, 600), obstructs=3)
    game = Game(env, True)
    while True:
        action = [0, 0]
        for event in pygame.event.get():
            if (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE) \
                    or event.type == pygame.QUIT:
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    action = [-3, 0]
                    env.radar.x = env.agent.position[0]+action[0]
                elif event.key == pygame.K_d:
                    action = [3, 0]
                    env.radar.x = env.agent.position[0] + action[0]
                elif event.key == pygame.K_w:
                    action = [0, -3]
                    env.radar.y = env.agent.position[1] + action[1]
                elif event.key == pygame.K_s:
                    action = [0, 3]
                    env.radar.y = env.agent.position[1] + action[1]
        action = np.array(action)
        s, r, info, _ = game.step(actions=action)
        print(r)
        game.render()
        print([i.position for i in env.obstructs],[i.d for i in env.obstructs])

if __name__ == '__main__':
    game_test()
#####################