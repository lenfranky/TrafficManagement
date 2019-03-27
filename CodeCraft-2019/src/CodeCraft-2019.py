# -*- encoding=utf8 -*-
# import logging
import sys


#
# logging.basicConfig(level=logging.DEBUG,
#                     filename=r'../logs/CodeCraft-2019.log',
#                     format='[%(asctime)s] %(levelname)s [%(funcName)s: %(filename)s, %(lineno)d] %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S',
#                     filemode='a')

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
        # print(self.road_id)
        # print(self.lanes_car_dict.items())


class Car(object):
    def __init__(self, params):
        # 车俩的id
        self.car_id = params[0]
        # 起始路口id
        self.start_crossing_id = params[1]
        # 重点路口id
        self.end_crossing_id = params[2]
        # 车辆的最高速度
        self.max_v = params[3]
        # 车辆的开始时间
        self.start_time = params[4]
        # 车辆的状态，有初始态I，冲突态C，等待态W，与终止态F
        # （删掉）两种设置方式，一种是0/1，0表示初始态与终止态，1表示等待态，每秒的第一次调度与之后的每次调度需要用两段代码
        # （删掉）另一种为0表示初始态，1表示等待态，2表示终止态，每秒调度完得恢复一次初始态
        self.status = 'I'

        # 堵到该车的车辆
        self.w_next_car = None
        # 该车所堵的所有车
        self.w_last_car = None

        # 该车在死锁检测中是否已被删除
        self.deleted_in_deadlock_detection = 0


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

        # 直接相邻的路口的对象
        self.up_crossing_object = None
        self.right_crossing_object = None
        self.down_crossing_object = None
        self.left_crossing_object = None

        # 所连接的下一路口的信息，格式为{next_crossing_id:[road_length, max_v, 0, road_id]} (注：倒数第二个0为测试用)
        self.next_crossing_dict = {}

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

        # 记录路口的连通性，结构为{crossing_id:{next_crossing_id:[road_length, road_max_v]}}
        self.crossing_connectivity_dict = {}

        # 每个时间点进入道路的车辆的索引，{时间点：车辆对象}
        self.enter_time_car_object_dict = {}

        # 读取文件
        self.load_file(road_info_file_path, car_info_file_path, crossing_info_file_path)

        # 建立对象之间的连接
        self.establish_link()

        # 建立路口连通性的索引
        self.init_crossing_connectivity()

        # 得到路口的矩阵
        self.crossing_array = self.get_crossing_array()

        # 得到每个路口上所发出的车的索引
        self.crossing_car_dict = self.get_crossing_car_dict()

        # 得到每个路口对应的在矩阵中的索引
        self.index_in_crossing_array = self.get_index_in_crossing_array()

        # 记录路线
        self.route_dict = {}

        # 记录中间节点的集合
        self.mid_node_set = set()

    def load_file(self, road_info_file_path, car_info_file_path, crossing_info_file_path):
        """
        读取文件中的信息
        :param road_info_file_path: 道路文件路径
        :param car_info_file_path: 车辆文件路径
        :param crossing_info_file_path: 路口文件路径
        :return:
        """
        # 读取道路信息文件
        with open(road_info_file_path, mode='r') as file_read:
            for line in file_read:
                if line[0] == '#' or len(line) == 0:
                    continue
                line = line.replace('(', '').replace(')', '').replace('\r', '').replace('\n', '')
                line_params = list(map(int, line.split(',')))
                # 若参数数量错误，则打印提示并跳过
                if len(line_params) != 7:
                    print("WRONG INPUT! -- Wrong road_info")
                    continue

                road_id = line_params[0]
                # 实例化道路对象
                road_object = Road(params=line_params)
                # 为道路对象建立索引
                self.road_id_to_object[road_id] = road_object

        # 读取车辆信息文件
        with open(car_info_file_path, mode='r') as file_read:
            for line in file_read:
                if line[0] == '#' or len(line) == 0:
                    continue
                line = line.replace('(', '').replace(')', '').replace('\r', '').replace('\n', '')
                line_params = list(map(int, line.split(',')))
                # 若参数数量错误，则打印提示并跳过
                if len(line_params) != 5:
                    print("WRONG INPUT! -- Wrong car_info")
                    continue

                car_id = line_params[0]
                # 实例化车辆对象
                car_object = Car(params=line_params)
                # 为车辆对象建立id索引
                self.car_id_to_object[car_id] = car_object
                # 为车辆建立进入时间的索引
                if line_params[4] in self.enter_time_car_object_dict:
                    self.enter_time_car_object_dict.get(line_params[4]).append(car_object)
                else:
                    self.enter_time_car_object_dict[line_params[4]] = [car_object]

        # 读取路口信息文件
        with open(crossing_info_file_path, mode='r') as file_read:
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

    def get_crossing_car_dict(self):
        crossing_car_dict = {}
        for car_id, car_object in self.car_id_to_object.items():
            if car_object.start_crossing_id in crossing_car_dict:
                crossing_car_dict[car_object.start_crossing_id].append(car_object)
            else:
                crossing_car_dict[car_object.start_crossing_id] = [car_object]
        return crossing_car_dict

    def has_no_deadlock_in_derict_graph(self):
        """
        检测是否有死锁
        :return: 如果没有死锁，则返回True
        """
        W_status_car_list = []
        for car_id in self.car_id_to_object:
            car_object = self.car_id_to_object[car_id]
            if car_object.status == 'W':
                W_status_car_list.append(car_object)

        in_dgree_dict = {}
        edge_array = []
        zero_in_degree_vertex_list = []
        zero_in_degree_vertex_count = []

        vertex_to_delete_list = []

        for car_object in W_status_car_list:
            if car_object.w_next_car.status != 'W':
                vertex_to_delete_list.append(car_object)

        return False

    def has_no_deadlock_in(self):
        """
        检测当前调度状态下是否有死锁
        :return: 若有死锁，则返回True
        """
        # 遍历所有车辆，将状态为W的车辆取出，并保存
        w_status_car_list = []
        for car_id in self.car_id_to_object:
            car_object = self.car_id_to_object[car_id]
            if car_object.status == 'W':
                w_status_car_list.append(car_object)

        # 遍历所有W节点的车辆，将没有堵住其他车辆的车找到并记录，作为root_node
        root_node_list = []
        for car_object in w_status_car_list:
            if car_object.w_last_car.status != 'W':
                root_node_list.append(car_object)

        # 通过root_node遍历每一个链表，在遍历的同时记录当前走过多少个节点
        # 若当前走过的节点数量大于当前的总的W状态车的数量，则存在死锁
        # 若一个链表被遍历完，则将总的W状态车的数量减去当前链表中走过的点的数量，并从下一个root_node开始遍历下一个链表
        total_list_length = len(w_status_car_list)
        for node in root_node_list:
            curr_count = 0
            while True:
                curr_count += 1
                if curr_count > total_list_length:
                    return True
                if node.w_next_car is None:
                    break
                else:
                    node = node.w_next_car
            total_list_length -= curr_count

        return False

    def init_crossing_connectivity(self):
        """
        建立路口间的连通性索引
        :return:
        """
        for crossing_id in self.crossing_id_to_object:
            self.crossing_connectivity_dict[crossing_id] = {}
        for crossing_id in self.crossing_id_to_object:
            crossing_object = self.crossing_id_to_object[crossing_id]
            # for road_object in [crossing_object.up_road_object, crossing_object.right_road_object,
            #                     crossing_object.down_road_object, crossing_object.left_road_object]:
            road_object = crossing_object.up_road_object
            next_crossing_object = None
            if road_object is not None:
                if road_object.start_crossing_id == crossing_id:
                    self.crossing_connectivity_dict[crossing_id][road_object.end_crossing_id] = [
                        road_object.road_length, road_object.max_v, 0]
                    crossing_object.next_crossing_dict[road_object.end_crossing_id] = [road_object.road_length,
                                                                                       road_object.max_v, 0,
                                                                                       road_object.road_id]
                    next_crossing_object = self.crossing_id_to_object[road_object.end_crossing_id]
                elif road_object.flag_is_two_way_road and road_object.end_crossing_id == crossing_id:
                    self.crossing_connectivity_dict[crossing_id][road_object.start_crossing_id] = [
                        road_object.road_length, road_object.max_v, 0]
                    crossing_object.next_crossing_dict[road_object.start_crossing_id] = [road_object.road_length,
                                                                                         road_object.max_v, 0,
                                                                                         road_object.road_id]
                    next_crossing_object = self.crossing_id_to_object[road_object.start_crossing_id]
                crossing_object.up_crossing_object = next_crossing_object

            road_object = crossing_object.right_road_object
            next_crossing_object = None
            if road_object is not None:
                if road_object.start_crossing_id == crossing_id:
                    self.crossing_connectivity_dict[crossing_id][road_object.end_crossing_id] = [
                        road_object.road_length, road_object.max_v, 0]
                    crossing_object.next_crossing_dict[road_object.end_crossing_id] = [road_object.road_length,
                                                                                       road_object.max_v, 0,
                                                                                       road_object.road_id]
                    next_crossing_object = self.crossing_id_to_object[road_object.end_crossing_id]
                elif road_object.flag_is_two_way_road and road_object.end_crossing_id == crossing_id:
                    self.crossing_connectivity_dict[crossing_id][road_object.start_crossing_id] = [
                        road_object.road_length, road_object.max_v, 0]
                    crossing_object.next_crossing_dict[road_object.start_crossing_id] = [road_object.road_length,
                                                                                         road_object.max_v, 0,
                                                                                         road_object.road_id]
                    next_crossing_object = self.crossing_id_to_object[road_object.start_crossing_id]
                crossing_object.right_crossing_object = next_crossing_object

            road_object = crossing_object.down_road_object
            next_crossing_object = None
            if road_object is not None:
                if road_object.start_crossing_id == crossing_id:
                    self.crossing_connectivity_dict[crossing_id][road_object.end_crossing_id] = [
                        road_object.road_length, road_object.max_v, 0]
                    crossing_object.next_crossing_dict[road_object.end_crossing_id] = [road_object.road_length,
                                                                                       road_object.max_v, 0,
                                                                                       road_object.road_id]
                    next_crossing_object = self.crossing_id_to_object[road_object.end_crossing_id]
                elif road_object.flag_is_two_way_road and road_object.end_crossing_id == crossing_id:
                    self.crossing_connectivity_dict[crossing_id][road_object.start_crossing_id] = [
                        road_object.road_length, road_object.max_v, 0]
                    crossing_object.next_crossing_dict[road_object.start_crossing_id] = [road_object.road_length,
                                                                                         road_object.max_v, 0,
                                                                                         road_object.road_id]
                    next_crossing_object = self.crossing_id_to_object[road_object.start_crossing_id]
                crossing_object.down_crossing_object = next_crossing_object

            road_object = crossing_object.left_road_object
            next_crossing_object = None
            if road_object is not None:
                if road_object.start_crossing_id == crossing_id:
                    self.crossing_connectivity_dict[crossing_id][road_object.end_crossing_id] = [
                        road_object.road_length, road_object.max_v, 0]
                    crossing_object.next_crossing_dict[road_object.end_crossing_id] = [road_object.road_length,
                                                                                       road_object.max_v, 0,
                                                                                       road_object.road_id]
                    next_crossing_object = self.crossing_id_to_object[road_object.end_crossing_id]
                elif road_object.flag_is_two_way_road and road_object.end_crossing_id == crossing_id:
                    self.crossing_connectivity_dict[crossing_id][road_object.start_crossing_id] = [
                        road_object.road_length, road_object.max_v, 0]
                    crossing_object.next_crossing_dict[road_object.start_crossing_id] = [road_object.road_length,
                                                                                         road_object.max_v, 0,
                                                                                         road_object.road_id]
                    next_crossing_object = self.crossing_id_to_object[road_object.start_crossing_id]
                crossing_object.left_crossing_object = next_crossing_object

    def find_shortest_node(self, cost, visited):
        """
        找到当前状态的最近未访问点
        :param cost:
        :param visited:
        :return:
        """
        min_dist = None
        node = None
        for i in self.crossing_id_to_object:
            if i in visited or cost[i] < 0:
                continue
            if (min_dist is None) or (cost[i] < min_dist):
                min_dist = cost[i]
                node = i
        # print (node)
        return node

    def find_min_time_path_with_dijkstra(self, car_object):
        """
        通过Dijkstra算法获取最短路径
        :param car_object:
        :return:
        """
        # car_object = Car(car_object)
        start_crossing_id = car_object.start_crossing_id
        end_crossing_id = car_object.end_crossing_id
        max_v = car_object.max_v

        for crossing_id in self.crossing_id_to_object:
            for next_crossing_id in self.crossing_id_to_object[crossing_id].next_crossing_dict:
                # print (next_crossing_id)
                curr_road_length = self.crossing_id_to_object[crossing_id].next_crossing_dict[next_crossing_id][0]
                curr_v = min(self.crossing_id_to_object[crossing_id].next_crossing_dict[next_crossing_id][1], max_v)
                # 更新当前的道路行驶所需时间
                self.crossing_id_to_object[crossing_id].next_crossing_dict[next_crossing_id][2] = int(
                    (curr_road_length + curr_v - 1) / curr_v)
        # dist = self.crossing_connectivity_dict

        # 由起点（结点1）到其余顶点的最短距离，-1代表无法到达
        cost = {}
        for crossing_id in self.crossing_id_to_object:
            cost[crossing_id] = -1
        cost[start_crossing_id] = 0
        for next_crossing_id in self.crossing_id_to_object[start_crossing_id].next_crossing_dict:
            cost[next_crossing_id] = self.crossing_id_to_object[start_crossing_id].next_crossing_dict[next_crossing_id][
                2]

        # parent代表到达这个结点的最短路径的前一个结点
        parents = {}
        for crossing_id in self.crossing_id_to_object:
            parents[crossing_id] = None
        parents[start_crossing_id] = None
        for crossing_id in self.crossing_id_to_object[car_object.start_crossing_id].next_crossing_dict:
            parents[crossing_id] = self.crossing_id_to_object[start_crossing_id]

        # 起始结点默认已经访问过
        visited = [start_crossing_id]

        # 更新最短路径
        node = self.find_shortest_node(cost, visited)
        while node:
            # print("node:\t" + str(node))
            curr_crossing_object = self.crossing_id_to_object[node]
            for i in curr_crossing_object.next_crossing_dict:  # 所有node结点的邻居结点
                new_cost = cost[node] + curr_crossing_object.next_crossing_dict[i][2]
                if cost[i] < 0 or new_cost < cost[i]:
                    # print(str(i) + "\t:\t" + str(cost[i]) + "\t:\t" + str(new_cost))
                    parents[i] = curr_crossing_object
                    # print(parents)
                    cost[i] = new_cost
            visited.append(node)
            node = self.find_shortest_node(cost, visited)

        min_time_route = []
        parent = parents[end_crossing_id]
        min_time_route.append(parent.next_crossing_dict[end_crossing_id][3])
        # print(parents)
        # print(min_time_route)

        # 不断向前索引，得到我们所需的路径
        curr_crossing_id = parent.crossing_id
        while True:
            # print(min_time_route)
            parent = parents[curr_crossing_id]
            if parent is None:
                break
            min_time_route.append(parent.next_crossing_dict[curr_crossing_id][3])
            curr_crossing_id = parent.crossing_id

        # 将前面倒序的列表做一次反转，并将start_node的None删去
        res = []
        # while len(min_time_route) > 0:
        #     curr_crossing_id = min_time_route.pop().crossing_id
        #     if curr_crossing_id:
        #         res.append(curr_crossing_id)
        #
        # print(cost[end_crossing_id])
        # print(res)

        # 将min_time_route进行反序排列
        res = min_time_route[::-1]

        # print(res)
        # print(cost[end_crossing_id])

        return cost[end_crossing_id], res

    def get_route_vertical(self, source_index, destination_index, is_frist_line_mid_node=False):
        """
        按照先纵向再横向的规则获得路线
        :param source_index:
        :param destination_index:
        :param is_frist_line_mid_node:
        :return:1)若找到了路线，则返回路线的list，其中每个元素为所走的道路的id
                2）若起始节点与目的节点位于同一行，但无法直接到达，则返回False
                3) 在其他情况下，若无法找到路线，则返回None
        """
        # print("%d -> %d" % (self.crossing_array[source_index[0]][source_index[1]].crossing_id, self.crossing_array[destination_index[0]][destination_index[1]].crossing_id))
        # 记录所走的路线
        route_list = []

        # 在遍历的过程中记录当前所在的行索引
        curr_row_index = source_index[0]
        # 在遍历的过程中记录当前所在的列索引
        curr_column_index = source_index[1]

        # 如果当前已经在同一行，则直接进行判断
        if source_index[0] == destination_index[0]:
            route_horizontal = self.get_route_horizontal(source_index, destination_index)
            # 如果获得了结果，则直接返回得到的路线
            if type(route_horizontal) is list:
                return route_horizontal
            # 否则，说明起始节点与目的节点在同一行，但并不能直达
            # 此时，返回False
            else:
                return False

        # 若目的节点在源节点的上方，即车辆需要先往下走
        if destination_index[0] > source_index[0]:
            # 获取当前节点，用于遍历
            curr_crossing_object = self.crossing_array[source_index[0]][source_index[1]]
            # 先不断向下遍历，知道到达目的行，或者不能再往下，记录当前的行索引
            while (curr_crossing_object is not None) and curr_row_index < destination_index[0]:
                route_list.append(curr_crossing_object.down_road_id)
                curr_crossing_object = curr_crossing_object.down_crossing_object
                if curr_crossing_object is None:
                    break
                curr_row_index += 1

            # 如果已经到达了目的节点所在的行，则再该行进行判断
            if curr_row_index == destination_index[0]:
                route_midnode_destination = self.get_route_horizontal((curr_row_index, source_index[1]), destination_index)
                # 如果在该行可以横向到达，则输出结果
                if type(route_midnode_destination) is list:
                    route_list.extend(route_midnode_destination)
                    return route_list
                curr_column_index = route_midnode_destination

            # 若在以上的几种情况下，都不能获得路线，则寻找中间节点进行辅助
            # 用迭代的方法来进行判断是否可以到达
            # 判断方式为：源节点是否能到达中间节点 AND 中间节点是否能到达目的节点
            # 而为了进行上面的两个判断，可能又需要更多的中间节点进行辅助

            # 首先需要获得中间点的选取范围
            # 若源节点在目的节点的右边，则列的选择的范围
            if source_index[1] >= destination_index[1]:
                start_column_index = destination_index[1]
                end_column_index = source_index[1] + 1
                column_range_list = list(range(start_column_index, end_column_index))
                column_range_list.reverse()
                rest_column_range_list = list(range(0, len(self.crossing_array[0])))
                rest_column_range_list.reverse()
            else:
                start_column_index = source_index[1]
                end_column_index = destination_index[1] + 1
                column_range_list = list(range(start_column_index, end_column_index))
                rest_column_range_list = list(range(0, len(self.crossing_array[0])))

            # print("%d -> %d" % (start_column_index, end_column_index))
            if is_frist_line_mid_node:
                row_range_list = list(range(source_index[0] + 1, curr_row_index + 1))
                row_range_list.reverse()
            else:
                row_range_list = list(range(source_index[0], curr_row_index + 1))
                row_range_list.reverse()


            # 不断地寻找中间点，并根据中间点进行判断
            for i in row_range_list:
                for j in column_range_list:
                    if self.can_reach_horizontal((i, source_index[1]), (i, j)):
                        pass
                    else:
                        continue
                    if i == source_index[0] and j == source_index[1]:
                        continue
                    mid_node_index = (i, j)
                    if mid_node_index in self.mid_node_set:
                        continue
                    else:
                        self.mid_node_set.add(mid_node_index)
                    # 若当前选择中间节点的行位于目的节点所在的行
                    if i == destination_index[0]:
                        route_midnode_destination = self.get_route_horizontal(mid_node_index, destination_index)
                        # 为了防止产生循环迭代，因此这里的第一个条件是用索引来辅助进行判断，而不是直接调用函数
                        # 因为既然i == destination_index[0]，说明已经到了目的节点所在的行
                        # 因此可以根据curr_column来进行判断
                        condition_1 = (source_index[1] < destination_index[1] and j < curr_column_index) or (
                                    source_index[1] > destination_index[1] and j > curr_column_index)
                        if condition_1 and (type(route_midnode_destination) is list):
                            route_from_source = self.get_route_vertical(source_index, mid_node_index, i == source_index[0])
                            route_from_source.extend(route_midnode_destination)
                            return route_from_source
                    else:
                        # print(self.crossing_array[i][j].crossing_id) i == source_index[0]
                        res_first = self.get_route_vertical(source_index, mid_node_index, i == source_index[0])
                        res_last = self.get_route_vertical(mid_node_index, destination_index, i == source_index[0])
                        if (type(res_first)is list) and (type(res_last)is list):
                            res_first.extend(res_last)
                            return res_first
                for j in rest_column_range_list:
                    # print("crossing_id:%d" % self.crossing_array[i][j].crossing_id)
                    if self.can_reach_horizontal((i, source_index[1]), (i, j)):
                        pass
                    else:
                        continue
                    if i == source_index[0] and j == source_index[1]:
                        continue
                    mid_node_index = (i, j)
                    if mid_node_index in self.mid_node_set:
                        continue
                    else:
                        self.mid_node_set.add(mid_node_index)
                    # 若当前选择中间节点的行位于目的节点所在的行
                    if i == destination_index[0]:
                        route_midnode_destination = self.get_route_horizontal(mid_node_index, destination_index)
                        # 为了防止产生循环迭代，因此这里的第一个条件是用索引来辅助进行判断，而不是直接调用函数
                        # 因为既然i == destination_index[0]，说明已经到了目的节点所在的行
                        # 因此可以根据curr_column来进行判断
                        condition_1 = (source_index[1] < destination_index[1] and j < curr_column_index) or (
                                    source_index[1] > destination_index[1] and j > curr_column_index)
                        if condition_1 and (type(route_midnode_destination) is list):
                            route_from_source = self.get_route_vertical(source_index, mid_node_index, i == source_index[0])
                            route_from_source.extend(route_midnode_destination)
                            return route_from_source
                    else:
                        # print(self.crossing_array[i][j].crossing_id)
                        res_first = self.get_route_vertical(source_index, mid_node_index, i == source_index[0])
                        res_last = self.get_route_vertical(mid_node_index, destination_index, i == source_index[0])
                        if (type(res_first)is list) and (type(res_last)is list):
                            res_first.extend(res_last)
                            return res_first

        # 若目的节点在源节点的下方，即车辆需要先往上走
        else:
            curr_crossing_object = self.crossing_array[source_index[0]][source_index[1]]
            while (curr_crossing_object is not None) and curr_row_index > destination_index[0]:
                route_list.append(curr_crossing_object.up_road_id)
                curr_crossing_object = curr_crossing_object.up_crossing_object
                if curr_crossing_object is None:
                    break
                curr_row_index -= 1

            if curr_row_index == destination_index[0]:
                route_midnode_destination = self.get_route_horizontal((curr_row_index, source_index[1]),
                                                                      destination_index)
                if type(route_midnode_destination) is list:
                    route_list.extend(route_midnode_destination)
                    return route_list
                curr_column_index = route_midnode_destination

            if source_index[1] >= destination_index[1]:
                start_column_index = destination_index[1]
                end_column_index = source_index[1] + 1
                column_range_list = list(range(start_column_index, end_column_index))
                column_range_list.reverse()
                rest_column_range_list = list(range(0, len(self.crossing_array[0])))
                rest_column_range_list.reverse()
            else:
                start_column_index = source_index[1]
                end_column_index = destination_index[1] + 1
                column_range_list = list(range(start_column_index, end_column_index))
                rest_column_range_list = list(range(0, len(self.crossing_array[0])))

            # print("%d - %d" % (start_column_index, end_column_index))
            if is_frist_line_mid_node:
                row_range_list = list(range(curr_row_index, source_index[0]))
            else:
                row_range_list = list(range(curr_row_index, source_index[0] + 1))
            for i in row_range_list:
                # print("i:%d" % i)
                for j in column_range_list:
                    # print(self.crossing_array[i][j].crossing_id)
                    if self.can_reach_horizontal((i, source_index[1]), (i, j)):
                        pass
                    else:
                        continue
                    if i == source_index[0] and j == source_index[1]:
                        continue
                    mid_node_index = (i, j)
                    if mid_node_index in self.mid_node_set:
                        continue
                    else:
                        self.mid_node_set.add(mid_node_index)
                    if i == destination_index[0]:
                        route_midnode_destination = self.get_route_horizontal(mid_node_index, destination_index)
                        condition_1 = (source_index[1] < destination_index[1] and j < curr_column_index) or (
                                    source_index[1] > destination_index[1] and j > curr_column_index)
                        if condition_1 and (type(route_midnode_destination) is list):
                            route_from_source = self.get_route_vertical(source_index, mid_node_index, i == source_index[0])
                            route_from_source.extend(route_midnode_destination)
                            return route_from_source
                    else:
                        # print(self.crossing_array[i][j].crossing_id)
                        res_first = self.get_route_vertical(source_index, mid_node_index, i == source_index[0])
                        res_last = self.get_route_vertical(mid_node_index, destination_index, i == source_index[0])
                        if (type(res_first)is list) and (type(res_last)is list):
                            res_first.extend(res_last)
                            return res_first
                for j in rest_column_range_list:
                    if self.can_reach_horizontal((i, source_index[1]), (i, j)):
                        pass
                    else:
                        continue
                    if i == source_index[0] and j == source_index[1]:
                        continue
                    mid_node_index = (i, j)
                    if mid_node_index in self.mid_node_set:
                        continue
                    else:
                        self.mid_node_set.add(mid_node_index)
                    if i == destination_index[0]:
                        route_midnode_destination = self.get_route_horizontal(mid_node_index, destination_index)
                        condition_1 = (source_index[1] < destination_index[1] and j < curr_column_index) or (
                                    source_index[1] > destination_index[1] and j > curr_column_index)
                        if condition_1 and (type(route_midnode_destination) is list):
                            route_from_source = self.get_route_vertical(source_index, mid_node_index, i == source_index[0])
                            route_from_source.extend(route_midnode_destination)
                            return route_from_source
                    else:
                        # print(self.crossing_array[i][j].crossing_id)
                        res_first = self.get_route_vertical(source_index, mid_node_index, i == source_index[0])
                        res_last = self.get_route_vertical(mid_node_index, destination_index, i == source_index[0])
                        if (type(res_first)is list) and (type(res_last)is list):
                            res_first.extend(res_last)
                            return res_first
        return 3

    def get_route_vertical_then_horizontal(self, car_object):
        self.mid_node_set.clear()
        source_index = self.index_in_crossing_array[self.crossing_id_to_object[car_object.start_crossing_id].crossing_id]
        destination_index = self.index_in_crossing_array[self.crossing_id_to_object[car_object.end_crossing_id].crossing_id]

        route = self.get_route_vertical(source_index=source_index, destination_index=destination_index)

        if type(route) is list:
            cost = self.get_route_cost_time(car_object, route_list=route)
            return cost, route
        else:
            return None

    def get_route_dict_list(self):
        route_dict_array = [[{}] * len(self.crossing_array[0]) for i in range(len(self.crossing_array))]

        width = len(self.crossing_array)
        hight = len(self.crossing_array[0])

        x_list = list(range(width))
        y_list = list(range(hight))
        x_list.reverse()
        y_list.reverse()

        for x in x_list:
            for y in y_list:
                source_crossing_object = self.crossing_array[x][y]

                i_list = list(range(x, width))
                j_list = list(range(y, hight))
                for i in i_list:
                    for j in j_list:
                        if i < x or j < y:
                            continue
                        route_dict_array[x][y][(i, j)] = self.get_route_vertical_priority((x, y), (i, j), route_dict_array)

    def get_route_vertical_priority(self, source_index, destination_index, route_dict_array):
        if source_index[0] <= destination_index[0]:
            pass
        return True

    def can_reach_horizontal(self, source_index, destination_index):
        if source_index[0] != destination_index[0]:
            print("Wrong Row Index!")
            return False
        if source_index[1] == destination_index[1]:
            return True

        if source_index[1] < destination_index[1]:
            curr_node = self.crossing_array[source_index[0]][source_index[1]]
            curr_column = source_index[1]
            while curr_column < destination_index[1] and curr_node is not None:
                curr_node = curr_node.right_crossing_object
                if curr_node is None:
                    break
                curr_column += 1

        else:
            curr_node = self.crossing_array[source_index[0]][source_index[1]]
            curr_column = source_index[1]
            while curr_column > destination_index[1] and curr_node is not None:
                curr_node = curr_node.left_crossing_object
                if curr_node is None:
                    break
                curr_column -= 1

        if curr_column == destination_index[1]:
            return True
        else:
            return False

    def get_route_horizontal(self, source_index, destination_index):
        route_list = []

        if source_index[1] == destination_index[1]:
            return []
        curr_column_index = source_index[1]
        curr_crossing_object = self.crossing_array[destination_index[0]][source_index[1]]
        if source_index[1] < destination_index[1]:
            while (curr_crossing_object is not None) and curr_column_index < destination_index[1]:
                route_list.append(curr_crossing_object.right_road_id)
                curr_crossing_object = curr_crossing_object.right_crossing_object
                if curr_crossing_object is None:
                    break
                curr_column_index += 1

        else:
            while (curr_crossing_object is not None) and curr_column_index > destination_index[1]:
                route_list.append(curr_crossing_object.left_road_id)
                curr_crossing_object = curr_crossing_object.left_crossing_object
                if curr_crossing_object is None:
                    break
                curr_column_index -= 1

        if curr_column_index == destination_index[1]:
            # route_list.pop()
            return route_list
        else:
            return curr_column_index

    def get_route_cost_time(self, car_object, route_list):
        """
        得到对于给定的车辆，走给定的路线需要花费的时间
        :param car_object:
        :param route_list:
        :return:
        """
        total_time = 0
        car_max_v = car_object.max_v

        for road_id in route_list:
            road_object = self.road_id_to_object[road_id]
            max_v = min(road_object.max_v, car_max_v)
            curr_time = int((road_object.road_length + max_v - 1) / max_v)
            total_time += curr_time

        return total_time

    def find_min_time_path_with_floyd(self):
        size = len(self.crossing_id_to_object)
        dis = [[-1] * size for i in range(size)]
        for crossing_id in self.crossing_id_to_object:
            crossing_object = self.crossing_id_to_object[crossing_id]

    def get_res(self, answer_path = 'answer.txt'):
        file_write = open(answer_path, mode='w')

        curr_time = 0
        for plan_time, car_object_list in self.enter_time_car_object_dict.items():
            if curr_time < plan_time:
                curr_time = plan_time
            for car_object in car_object_list:
                file_write.write("(%d, %d" % (car_object.car_id, curr_time))
                cost, min_time_route = self.find_min_time_path_with_dijkstra(car_object)
                curr_time += cost
                for road_id in min_time_route:
                    file_write.write(", %d" % road_id)
                file_write.write(")\n")

        file_write.close()

    def get_res_per_crossing(self, answer_path = 'answer.txt'):
        file_write = open(answer_path, mode='w')

        start_time = 0
        for car_id, car_object in self.car_id_to_object.items():
            if car_object.start_time > start_time:
                start_time = car_object.start_time

        crossing_car_dict = {}
        for car_id, car_object in self.car_id_to_object.items():
            if car_object.start_crossing_id in crossing_car_dict:
                crossing_car_dict[car_object.start_crossing_id].append(car_object)
            else:
                crossing_car_dict[car_object.start_crossing_id] = [car_object]

        car_num_list = []
        for crossing_id, car_object_list in crossing_car_dict.items():
            car_num_list.append(len(car_object_list))
        # sorted_car_num_list = sorted(car_num_list)

        curr_time = start_time
        for crossing_id, car_object_list in crossing_car_dict.items():
            max_time = 0
            for car_object in car_object_list:
                cost, min_time_route = self.find_min_time_path_with_dijkstra(car_object)
                if cost > max_time:
                    max_time = cost
                file_write.write("(%d, %d" % (car_object.car_id, curr_time))
                for road_id in min_time_route:
                    file_write.write(", %d" % road_id)
                file_write.write(")\n")
            curr_time += max_time / 1

        file_write.close()

    def get_res_per_crossing_line(self, answer_path='answer.txt'):
        file_write = open(answer_path, mode='w')

        start_time = 0
        for car_id, car_object in self.car_id_to_object.items():
            if car_object.start_time > start_time:
                start_time = car_object.start_time

        curr_time = start_time

        size = len(self.crossing_array)
        line_index_list = [0] * size

        mid = int(size / 2)
        if size % 2 == 1:
            mid += 1

        list_1 = list(range(0, mid))
        list_2 = list(range(mid, size))

        index = 0
        for num in list_1:
            line_index_list[index] = num
            index += 2

        index = 1
        for num in list_2:
            line_index_list[index] = num
            index += 2

        # print(line_index_list)

        for line_index in line_index_list:
            crossing_line = self.crossing_array[line_index]
            max_time_cost_line = 0
            for crossing_object in crossing_line:
                arranged_car_list = []
                for car_object in self.crossing_car_dict[crossing_object.crossing_id]:
                    res = self.get_route_vertical_then_horizontal(car_object)
                    if res is None:
                        continue
                    cost, route_list = res
                    arranged_car_list.append(car_object)
                    if cost > max_time_cost_line:
                        max_time_cost_line = cost
                    file_write.write("(%d, %d" % (car_object.car_id, curr_time))
                    for road_id in route_list:
                        file_write.write(", %d" % road_id)
                    file_write.write(")\n")

                arranged_car_list = set(arranged_car_list)
                for car_object in arranged_car_list:
                    self.crossing_car_dict[crossing_object.crossing_id].remove(car_object)
                    self.car_id_to_object.pop(car_object.car_id)
            curr_time += max_time_cost_line * 1 / 1

        # curr_time = start_time
        for crossing_id, car_object_list in self.crossing_car_dict.items():
            max_time = 0
            for car_object in car_object_list:
                cost, min_time_route = self.find_min_time_path_with_dijkstra(car_object)
                if cost > max_time:
                    max_time = cost
                file_write.write("(%d, %d" % (car_object.car_id, curr_time))
                for road_id in min_time_route:
                    file_write.write(", %d" % road_id)
                file_write.write(")\n")
            curr_time += max_time * 1 / 1

        file_write.close()

    def get_res_per_crossing_line_prj(self, answer_path='answer.txt'):
        file_write = open(answer_path, mode='w')

        start_time = 0
        for car_id, car_object in self.car_id_to_object.items():
            if car_object.start_time > start_time:
                start_time = car_object.start_time

        curr_time = start_time

        size = len(self.crossing_array)
        line_index_list = [0] * size

        mid = int(size / 2)
        if size % 2 == 1:
            mid += 1

        list_1 = list(range(0, mid))
        list_2 = list(range(mid, size))

        index = 0
        for num in list_1:
            line_index_list[index] = num
            index += 2

        index = 1
        for num in list_2:
            line_index_list[index] = num
            index += 2

        # print(line_index_list)

        for line_index in line_index_list:
            crossing_line = self.crossing_array[line_index]
            max_time_cost_line = 0
            for crossing_object in crossing_line:
                arranged_car_list = []
                for car_object in self.crossing_car_dict[crossing_object.crossing_id]:
                    res = self.get_route_vertical_then_horizontal(car_object)
                    if res is None:
                        continue
                    cost, route_list = res
                    arranged_car_list.append(car_object)
                    if cost > max_time_cost_line:
                        max_time_cost_line = cost
                    file_write.write("(%d, %d" % (car_object.car_id, curr_time))
                    for road_id in route_list:
                        file_write.write(", %d" % road_id)
                    file_write.write(")\n")

                arranged_car_list = set(arranged_car_list)
                for car_object in arranged_car_list:
                    self.crossing_car_dict[crossing_object.crossing_id].remove(car_object)
                    self.car_id_to_object.pop(car_object.car_id)
            curr_time += max_time_cost_line * 3 / 4

        curr_time = start_time
        # 通过遍历两遍来实现相应功能，第一遍遍历找到偶数坐标点所需要的最大时间，第二遍遍历实现写入txt
        even_max_time = 0
        odd_max_time = 0
        for crossing_id, car_object_list in self.crossing_car_dict.items():
            # 通过crossing_id获取横纵坐标
            [x, y] = self.index_in_crossing_array[crossing_id]

            for car_object in car_object_list:
                cost, min_time_route = self.find_min_time_path_with_dijkstra(car_object)
                if (cost > even_max_time) and (x + y) % 2 == 0:
                    even_max_time = cost
                    continue
                if (cost > odd_max_time) and (x + y) % 2 != 0:
                    odd_max_time = cost

        for crossing_id, car_object_list in self.crossing_car_dict.items():
            [x, y] = self.index_in_crossing_array[crossing_id]
            if (x + y) % 2 == 0:
                road_time = even_max_time + curr_time
            if (x + y) % 2 != 0:
                road_time = even_max_time + odd_max_time + curr_time
            for car_object in car_object_list:
                cost, min_time_route = self.find_min_time_path_with_dijkstra(car_object)

                file_write.write("(%d, %d" % (car_object.car_id, road_time))
                for road_id in min_time_route:
                    file_write.write(", %d" % road_id)
                file_write.write(")\n")

        file_write.close()

    def get_res_per_direction(self, answer_path='answer.txt'):
        file_write = open(answer_path, mode='w')

        # 起始点位于终点上方的车辆的列表(或两者位于同一行)
        above_car_list = []
        # 起始点位于终点下方的车辆的列表
        below_car_list = []

        start_time = 0
        for car_id, car_object in self.car_id_to_object.items():
            if car_object.start_time > start_time:
                start_time = car_object.start_time

        for car_id, car_object in self.car_id_to_object.items():
            start_index = self.index_in_crossing_array[car_object.start_crossing_id]
            end_index = self.index_in_crossing_array[car_object.end_crossing_id]

            if start_index[0] <= end_index[0]:
                above_car_list.append((car_id, car_object))
            else:
                below_car_list.append((car_id, car_object))

        curr_time = start_time
        arranged_car_list = []

        max_time_below_cars = 0
        for car_id, car_object in below_car_list:
            res = self.get_route_vertical_then_horizontal(car_object)
            if res is None:
                continue
            cost, route_list = res
            arranged_car_list.append(car_object)
            if cost > max_time_below_cars:
                max_time_below_cars = cost
            file_write.write("(%d, %d" % (car_object.car_id, curr_time))
            for road_id in route_list:
                file_write.write(", %d" % road_id)
            file_write.write(")\n")

        curr_time += max_time_below_cars * 1

        max_time_above_cars = 0
        for car_id, car_object in above_car_list:
            res = self.get_route_vertical_then_horizontal(car_object)
            if res is None:
                continue
            cost, route_list = res
            arranged_car_list.append(car_object)
            if cost > max_time_above_cars:
                max_time_above_cars = cost
            file_write.write("(%d, %d" % (car_object.car_id, curr_time))
            for road_id in route_list:
                file_write.write(", %d" % road_id)
            file_write.write(")\n")

        curr_time += max_time_above_cars * 1

        for car_object in arranged_car_list:
            self.car_id_to_object.pop(car_object.car_id)

        # 更新每个路口所发出的车的索引
        self.crossing_car_dict = self.get_crossing_car_dict()

        car_count_second = 0
        for crossing_id, car_object_list in self.crossing_car_dict.items():
            max_time = 0
            for car_object in car_object_list:
                car_count_second += 1
                cost, min_time_route = self.find_min_time_path_with_dijkstra(car_object)
                if cost > max_time:
                    max_time = cost
                file_write.write("(%d, %d" % (car_object.car_id, curr_time))
                for road_id in min_time_route:
                    file_write.write(", %d" % road_id)
                file_write.write(")\n")
            curr_time += max_time

        file_write.close()

        # print(car_count_second)

    def get_root_node(self):
        root_node = None
        root_node_candidate = []
        for crossing_id, crossing_object in self.crossing_id_to_object.items():
            if crossing_object.up_road_object is None and crossing_object.left_road_object is None:
                root_node_candidate.append(crossing_object)

        if len(root_node_candidate) > 1:
            delete_node_list = []
            for crossing_object in root_node_candidate:
                curr_crossing = crossing_object
                while curr_crossing is not None:
                    curr_crossing = curr_crossing.down_road_object
                    if curr_crossing.left_road_object is not None:
                        delete_node_list.append(crossing_object)
                        break
                if crossing_object in delete_node_list:
                    continue
                curr_crossing = crossing_object
                while curr_crossing is not None:
                    curr_crossing = curr_crossing.right_road_object
                    if curr_crossing.up_road_object is not None:
                        delete_node_list.append(crossing_object)
                        break
            for crossing_object in root_node_candidate:
                if crossing_object not in delete_node_list:
                    root_node = crossing_object

        else:
            root_node = root_node_candidate[0]

        return root_node

    def show_crossing_array(self, crossing_array):
        for i in range(len(crossing_array)):
            for j in range(len(crossing_array[0])):
                if crossing_array[i][j] is not None:
                    print(crossing_array[i][j].crossing_id, end='\t')
                else:
                    print("N", end='\t')
            print()


    def get_index_from_input_array(self, target_crossing_object, crossing_array):
        if target_crossing_object is None:
            return None
        target_crossing_id = target_crossing_object.crossing_id
        for i in range(len(crossing_array)):
            for j in range(len(crossing_array[0])):
                if crossing_array[i][j] is None:
                    continue
                if crossing_array[i][j].crossing_id == target_crossing_id:
                    return [i, j]

        return None

    def get_index_in_crossing_array(self):
        """
        得到每个路口在路口的矩阵中的索引的map
        :return: crossing_index_in_crossing_array
        """
        crossing_index_in_crossing_array = {}

        for i in range(len(self.crossing_array)):
            for j in range(len(self.crossing_array[0])):
                crossing_index_in_crossing_array[self.crossing_array[i][j].crossing_id] = (i, j)

        return crossing_index_in_crossing_array


    def get_crossing_array(self):
        """
        获取一个矩阵，其中按照路口的实际位置记录所有的路口
        :return:
        """
        root_node = self.get_root_node()
        column_count = 0
        row_count = 0
        first_row_node_list = []
        first_column_node_list = []
        node_placed = []

        curr_node = root_node
        while curr_node is not None:
            first_row_node_list.append(curr_node)
            curr_node = curr_node.left_crossing_object

        curr_node = root_node
        while curr_node is not None:
            first_column_node_list.append(curr_node)
            curr_node = curr_node.down_crossing_object

        for first_node in first_row_node_list:
            curr_node = first_node
            count = 0
            while curr_node is not None:
                curr_node = curr_node.down_crossing_object
                count += 1
            if count > row_count:
                row_count = count

        for first_node in first_column_node_list:
            curr_node = first_node
            count = 0
            while curr_node is not None:
                curr_node = curr_node.right_crossing_object
                count += 1
            if count > column_count:
                column_count = count

        crossing_array = [[None] * column_count for i in range(row_count)]
        # print(crossing_array)
        crossing_array[0][0] = root_node

        node_placed.append(root_node.crossing_id)
        for i in range(1, len(first_row_node_list)):
            crossing_array[0][i] = first_row_node_list[i]
            node_placed.append(first_row_node_list[i].crossing_id)

        for i in range(1, len(first_column_node_list)):
            crossing_array[i][0] = first_column_node_list[i]
            node_placed.append(first_column_node_list[i].crossing_id)

        curr_column = 0
        for first_node in first_row_node_list:
            curr_row = 1
            curr_node = first_node.down_crossing_object
            while curr_node is not None:
                # print("%d %d" % (curr_row, curr_column))
                crossing_array[curr_row][curr_column] = curr_node
                node_placed.append(curr_node.crossing_id)
                curr_node = curr_node.down_crossing_object
                curr_row += 1
            curr_column += 1

        curr_row = 0
        for first_node in first_column_node_list:
            curr_column = 1
            curr_node = first_node.right_crossing_object
            while curr_node is not None:
                # print("%d %d" % (curr_row, curr_column))
                crossing_array[curr_row][curr_column] = curr_node
                node_placed.append(curr_node.crossing_id)
                curr_node = curr_node.right_crossing_object
                curr_column += 1
            curr_row += 1

        node_placed = set(node_placed)

        node_not_placed = []

        for crossing_id, crossing_object in self.crossing_id_to_object.items():
            if crossing_id not in node_placed:
                node_not_placed.append(crossing_object)

        while len(node_not_placed) > 0:
            node_placed = []
            for crossing_object in node_not_placed:
                flag_find_index = False
                # 如果上方节点已被放置
                if not flag_find_index:
                    neighbor_index = self.get_index_from_input_array(
                        target_crossing_object=crossing_object.up_crossing_object,
                        crossing_array=crossing_array)
                    if neighbor_index is not None:
                        node_placed.append(crossing_object)
                        crossing_array[neighbor_index[0] + 1][neighbor_index[1]] = crossing_object
                        flag_find_index = True

                # 如果左方节点已被放置
                if not flag_find_index:
                    neighbor_index = self.get_index_from_input_array(
                        target_crossing_object=crossing_object.left_crossing_object,
                        crossing_array=crossing_array)
                    if neighbor_index is not None:
                        node_placed.append(crossing_object)
                        crossing_array[neighbor_index[0]][neighbor_index[1] + 1] = crossing_object
                        flag_find_index = True

                # 如果右方节点已被放置
                if not flag_find_index:
                    neighbor_index = self.get_index_from_input_array(
                        target_crossing_object=crossing_object.right_crossing_object,
                        crossing_array=crossing_array)
                    if neighbor_index is not None:
                        node_placed.append(crossing_object)
                        crossing_array[neighbor_index[0]][neighbor_index[1] - 1] = crossing_object
                        flag_find_index = True

                # 如果下方节点已被放置
                if not flag_find_index:
                    neighbor_index = self.get_index_from_input_array(
                        target_crossing_object=crossing_object.up_crossing_object,
                        crossing_array=crossing_array)
                    if neighbor_index is not None:
                        node_placed.append(crossing_object)
                        crossing_array[neighbor_index[0] - 1][neighbor_index[1]] = crossing_object
                        flag_find_index = True

            for node in node_placed:
                node_not_placed.remove(node)

        return crossing_array

    def get_res_per_line_with_min_time_cost_path(self, answer_path = 'answer.txt'):
        """
        以迪杰斯特拉算法获取每辆车从起始路口到终点路口的路径，每次发车的时候，发一整行的路口
        :param answer_path:
        :return:
        """
        file_write = open(answer_path, mode='w')

        start_time = 0
        for car_id, car_object in self.car_id_to_object.items():
            if car_object.start_time > start_time:
                start_time = car_object.start_time

        crossing_car_dict = {}
        for car_id, car_object in self.car_id_to_object.items():
            if car_object.start_crossing_id in crossing_car_dict:
                crossing_car_dict[car_object.start_crossing_id].append(car_object)
            else:
                crossing_car_dict[car_object.start_crossing_id] = [car_object]

        crossing_array = self.get_crossing_array()

        curr_time = start_time
        for crossing_object_list in crossing_array:
            max_time = 0
            for crossing_object in crossing_object_list:
                car_object_list = crossing_car_dict[crossing_object.crossing_id]
                for car_object in car_object_list:
                    cost, min_time_route = self.find_min_time_path_with_dijkstra(car_object)
                    if cost > max_time:
                        max_time = cost
                    file_write.write("(%d, %d" % (car_object.car_id, curr_time))
                    for road_id in min_time_route:
                        file_write.write(", %d" % road_id)
                    file_write.write(")\n")
            curr_time += max_time

    def get_res_per_line(self, answer_path = 'answer.txt'):
        file_write = open(answer_path, mode='w')
        self.show_crossing_array(self.crossing_array)


def main():
    # if len(sys.argv) != 5:
    #     logging.info('please input args: car_path, road_path, cross_path, answerPath')
    #     exit(1)

    car_path = sys.argv[1]
    road_path = sys.argv[2]
    cross_path = sys.argv[3]
    answer_path = sys.argv[4]

    tm = TrafficManaging(road_path, car_path, cross_path)
    tm.get_res_per_crossing_line_prj(answer_path)

    # logging.info("car_path is %s" % (car_path))
    # logging.info("road_path is %s" % (road_path))
    # logging.info("cross_path is %s" % (cross_path))
    # logging.info("answer_path is %s" % (answer_path))


# to read input file
# process
# to write output file


if __name__ == "__main__":
    main()
