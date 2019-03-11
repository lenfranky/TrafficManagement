# -*- coding: utf-8 -*-
class Road(object):
    def __init__(self, params):
        # 道路id
        self.road_id = params[0]
        # 道路长度
        self.road_length = params[1]
        # 最高速度
        self.max_v = params[2]
        # 车道的数目
        self.lanes_num = params[3]
        # 起始路口的id
        self.start_crossing_id = params[4]
        # 终点路口的id
        self.end_crossing_id = params[5]
        # 是否为双向的标志（1：双向，0：单向）
        self.flag_is_two_way_road = params[6]

        # 起始点路口的对象
        self.start_crossing_object = None
        # 重点路口的对象
        self.end_crossing_object = None

    def link_to_crossing_object(self, crossing_id_to_object):
        """
        与路口的对象进行链接
        :param crossing_id_to_object: 路口id到对象的索引
        :return:
        """
        self.start_crossing_object = crossing_id_to_object[self.start_crossing_id]
        self.end_crossing_object = crossing_id_to_object[self.end_crossing_id]


class Car(object):
    def __init__(self, params):
        # 车俩的id
        self.car_id = params[0]
        # 起始路口id
        self.start_crossing_is = params[1]
        # 重点路口id
        self.end_crossing_id = params[2]
        # 车辆的最高速度
        self.max_v = params[3]
        # 车辆的开始时间
        self.start_time = params[4]


class Crossing(object):
    def __init__(self, params):
        # 路口的id
        self.crossing_id = params[0]
        # 顺时针排列的四个道路的id，以向上的方向开始
        self.up_road_id = params[1]
        self.right_road_id = params[2]
        self.down_road_id = params[3]
        self.left_road_id = params[4]

        # 所链接的道路的对象
        self.up_road_object = None
        self.right_road_object = None
        self.down_road_object = None
        self.left_road_object = None

    def link_to_road_object(self, road_id_to_object):
        """
        与道路的对象建立链接
        :param road_id_to_object:道路id到对象的索引
        :return:
        """
        self.up_road_object = road_id_to_object[self.up_road_id] if self.up_road_id != -1 else None
        self.right_road_object = road_id_to_object[self.right_road_id] if self.right_road_id != -1 else None
        self.down_road_object = road_id_to_object[self.down_road_id] if self.down_road_id != -1 else None
        self.left_road_object = road_id_to_object[self.left_road_id] if self.left_road_id != -1 else None

