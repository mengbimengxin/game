import numpy as np


# class Get_State(object):
#     def __init__(self, Agent):
#         self.speed = Agent.speed
#         self.color = Agent.color
#
#     def unit_speed(self):
#         return self.speed / (np.linalg.norm(self.speed))
#
#     def vertical_speed(self, agent):
#         if self.color == "r":
#             v_s1 = np.array(self.speed[0] * np.cos(np.pi / 2) - self.speed[1] * np.sin(np.pi / 2),
#                             self.speed[1] * np.cos(np.pi / 2) + self.speed[0] * np.sin(np.pi / 2))
#             v_s2 = np.array(self.speed[0] * np.cos(-np.pi / 2) - self.speed[1] * np.sin(-np.pi / 2),
#                             self.speed[1] * np.cos(-np.pi / 2) + self.speed[0] * np.sin(-np.pi / 2))
#
#             cos1 = np.sum(agent.speed * v_s1) / (np.linalg.norm(agent.speed) * np.linalg.norm(v_s1))
#             cos2 = np.sum(agent.speed * v_s2) / (np.linalg.norm(agent.speed) * np.linalg.norm(v_s2))
#             if cos1 >= 0 and cos2 < 0:
#                 print(v_s1)
#                 return v_s1
#             elif cos2 >= 0 and cos2 < 0:
#                 print(v_s2)
#                 return v_s2


def get_state(agent_speed: np.ndarray, obstructure_speed: np.ndarray):
    # 单位速度
    a_speed = agent_speed / (np.linalg.norm(agent_speed))
    o_speed = obstructure_speed / (np.linalg.norm(obstructure_speed))
    # print(a_speed,o_speed)
    # 向量旋转
    v_s1 = np.asarray([[np.cos(np.pi / 2), -np.sin(np.pi / 2)], [np.sin(np.pi / 2), np.cos(np.pi / 2)]]).dot(
        obstructure_speed.T)

    v_s2 = np.asarray([[np.cos(-np.pi / 2), -np.sin(-np.pi / 2)], [np.sin(-np.pi / 2), np.cos(-np.pi / 2)]]).dot(
        obstructure_speed.T)
    # print(v_s1, v_s2)
    # 旋转向量与agent速度向量的夹角
    cos1 = np.sum(agent_speed * v_s1.T) / (np.linalg.norm(agent_speed) * np.linalg.norm(v_s1))
    cos2 = np.sum(agent_speed * v_s2.T) / (np.linalg.norm(agent_speed) * np.linalg.norm(v_s2))
    round_ = np.vectorize(lambda x: round(x, 2))
    #判断哪个夹角为锐角
    if cos1 >= 0 and cos2 < 0:
        print("垂直向量:",round_(v_s1),"agent单位速度:", round_(a_speed), "obstructure单位速度:",round_(o_speed))
        return round_(v_s1), round_(a_speed), round_(o_speed)
    elif cos2 >= 0 and cos1 < 0:
        print("垂直向量:",round_(v_s2),"agent单位速度:", round_(a_speed), "obstructure单位速度:",round_(o_speed))
        return round_(v_s2), round_(a_speed), round_(o_speed)


get_state(agent_speed=np.asarray([1, 1]), obstructure_speed=np.asarray([0, 1]))
