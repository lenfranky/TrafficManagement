#-*- encoding=utf8 -*-
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
            for road_object in [crossing_object.up_road_object, crossing_object.right_road_object,
                                crossing_object.down_road_object, crossing_object.left_road_object]:
                # road_object = Road(road_object)
                if road_object is None:
                    continue
                if road_object.start_crossing_id == crossing_id:
                    self.crossing_connectivity_dict[crossing_id][road_object.end_crossing_id] = [
                        road_object.road_length, road_object.max_v, 0]
                    crossing_object.next_crossing_dict[road_object.end_crossing_id] = [road_object.road_length,
                                                                                       road_object.max_v, 0,
                                                                                       road_object.road_id]
                elif road_object.flag_is_two_way_road and road_object.end_crossing_id == crossing_id:
                    self.crossing_connectivity_dict[crossing_id][road_object.start_crossing_id] = [
                        road_object.road_length, road_object.max_v, 0]
                    crossing_object.next_crossing_dict[road_object.start_crossing_id] = [road_object.road_length,
                                                                                         road_object.max_v, 0,
                                                                                         road_object.road_id]

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


def main():
    # if len(sys.argv) != 5:
    #     logging.info('please input args: car_path, road_path, cross_path, answerPath')
    #     exit(1)

    car_path = sys.argv[1]
    road_path = sys.argv[2]
    cross_path = sys.argv[3]
    answer_path = sys.argv[4]

    tm = TrafficManaging(road_path, car_path, cross_path)
    tm.get_res(answer_path)

    # logging.info("car_path is %s" % (car_path))
    # logging.info("road_path is %s" % (road_path))
    # logging.info("cross_path is %s" % (cross_path))
    # logging.info("answer_path is %s" % (answer_path))

# to read input file
# process
# to write output file


if __name__ == "__main__":
    main()
