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
        # 终点路口的对象
        self.end_crossing_object = None

        # 当前时刻各个车道上的车辆{lanes_id: [car_objects]}
        self.lanes_car_dict = {}
        # 进行车道的初始化
        self.lanes_initialization()

    def link_to_crossing_object(self, crossing_id_to_object):
        """
        与路口的对象进行链接
        :param crossing_id_to_object: 路口id到对象的索引
        :return:
        """
        self.start_crossing_object = crossing_id_to_object[self.start_crossing_id]
        self.end_crossing_object = crossing_id_to_object[self.end_crossing_id]

    def lanes_initialization(self):
        """
        创建道路中的车道
        :return:
        """
        # 车道编号为从1到n，与从-1到-n
        for lane_id in range(1, self.lanes_num + 1):
            self.lanes_car_dict[lane_id] = Lane(lane_id=lane_id, length=self.road_length, max_v=self.max_v)
        if self.flag_is_two_way_road:
            for lane_id in range(1, self.lanes_num + 1):
                self.lanes_car_dict[-lane_id] = Lane(lane_id=lane_id, length=self.road_length, max_v=self.max_v)
        print(self.road_id)
        print(self.lanes_car_dict.items())


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
        # 车辆的状态，有初始态，等待态，与终止态
        # 两种设置方式，一种是0/1，0表示初始态与终止态，1表示等待态，每秒的第一次调度与之后的每次调度需要用两段代码
        # 另一种为0表示初始态，1表示等待态，2表示终止态，每秒调度完得恢复一次初始态
        self.status = 0


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


class Lane(object):
    """
    用来描述车道的类，内部用列表来实现一个队列
    """
    def __init__(self, lane_id, length, max_v):
        # 车道的id
        self.lane_id = lane_id
        # 车道的长度限制
        self.size = length
        # 车道上的最高速度
        self.max_v = max_v
        # 车辆对象的数组
        self.queue = []

    def enter_lane(self, car_object):
        """
        添加一个车辆对象并返回True，如果队列已满，则返回False
        :param car_object:
        :return:
        """
        if self.is_full():
            return False
        self.queue.append(car_object)
        return True

    def out_lane(self):
        """
        移除并返问队列头部的车辆对象，如果队列为空，则返回None
        :return:
        """
        if self.is_empty():
            return None
        return self.queue.pop(0)

    def get_first_car_object(self):
        """
        返回队列头部的车辆对象，如果队列为空，则返回None
        :return:
        """
        if self.is_empty():
            return None
        return self.queue[0]

    def is_empty(self):
        return len(self.queue) == 0

    def is_full(self):
        return len(self.queue) >= self.size

    def show_queue(self):
        print(self.queue)


class TrafficManaging(object):
    def __init__(self, road_info_file_path, car_info_file_path, crossing_info_file_path):
        # 由道路id到道路对象的索引{road_id：道路对象}
        self.road_id_to_object = {}
        # 由车辆id到车辆对象的索引
        self.car_id_to_object = {}
        # 由路口id到路口对象的索引
        self.crossing_id_to_object = {}

        # 每个时间点进入道路的车辆的索引，{时间点：车辆对象}
        self.enter_time_car_object_dict = {}

        # 读取文件
        self.load_file(road_info_file_path, car_info_file_path, crossing_info_file_path)
        # 建立对象之间的连接
        self.establish_link()

    def load_file(self, road_info_file_path, car_info_file_path, crossing_info_file_path):
        """
        读取文件中的信息
        :param road_info_file_path: 道路文件路径
        :param car_info_file_path: 车辆文件路径
        :param crossing_info_file_path: 路口文件路径
        :return:
        """
        # 读取道路信息文件
        with open(road_info_file_path, mode='r', encoding='utf-8') as file_read:
            for line in file_read:
                if line[0] == '#' or len(line) == 0:
                    continue
                line = line.replace('(', '').replace(')', '').replace('\r', '').replace('\n', '')
                line_params = list(map(int, line.split(',')))
                # 若参数数量错误，则打印提示并跳过
                if len(line_params) != 7:
                    print("WRONG INPUT!")
                    continue

                road_id = line_params[0]
                # 实例化道路对象
                road_object = Road(params=line_params)
                # 为道路对象建立索引
                self.road_id_to_object[road_id] = road_object

        # 读取车辆信息文件
        with open(car_info_file_path, mode='r', encoding='utf-8') as file_read:
            for line in file_read:
                if line[0] == '#' or len(line) == 0:
                    continue
                line = line.replace('(', '').replace(')', '').replace('\r', '').replace('\n', '')
                line_params = list(map(int, line.split(',')))
                # 若参数数量错误，则打印提示并跳过
                if len(line_params) != 5:
                    print("WRONG INPUT!")
                    continue

                car_id = line_params[0]
                # 实例化车辆对象
                car_object = Car(params=line_params)
                # 为车辆对象建立id索引
                self.car_id_to_object[car_id] = car_object
                # 为车辆建立进入时间的索引
                self.enter_time_car_object_dict.get(line_params[4], []).append(car_object)

        # 读取路口信息文件
        with open(crossing_info_file_path, mode='r', encoding='utf-8') as file_read:
            for line in file_read:
                if line[0] == '#' or len(line) == 0:
                    continue
                line = line.replace('(', '').replace(')', '').replace('\r', '').replace('\n', '')
                line_params = list(map(int, line.split(',')))
                # 若参数数量错误，则打印提示并跳过
                if len(line_params) != 5:
                    print("WRONG INPUT!")
                    continue

                crossing_id = line_params[0]
                # 实例化路口对象
                crossing_object = Crossing(params=line_params)
                # 为路口对象建立索引
                self.crossing_id_to_object[crossing_id] = crossing_object
        # # 标-1的路口为空
        # self.crossing_id_to_object[-1] = None

    def establish_link(self):
        for road_id, road_object in self.road_id_to_object.items():
            road_object.link_to_crossing_object(self.crossing_id_to_object)

        for crossing_id, crossing_object in self.crossing_id_to_object.items():
            crossing_object.link_to_road_object(self.road_id_to_object)


if __name__ == '__main__':
    config_num = 1
    tm = TrafficManaging('../config_%d/road.txt' % config_num, '../config_%d/car.txt' % config_num,
                         '../config_%d/cross.txt' % config_num)
