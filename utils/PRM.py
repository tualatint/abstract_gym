import numpy as np
import __init__
from abstract_gym.scenario.scene_2 import Scene
import matplotlib.pyplot as plt
from abstract_gym.environment.occupancy_grid import OccupancyGrid
import time
import random

class Vertex:
    def __init__(self, j1, j2):
        self.j1 = j1
        self.j2 = j2
        self.father = None
        self.g = np.inf  # The G value is the cost of the cheapest path from start to node n currently known.
        self.h = np.inf  # The Heuristic value estimate the distance between the node n and the goal

    def setFather(self, node):
        self.father = node

    def setG(self, g):
        self.g = g

    def setH(self, h):
        self.h = h

    def getF(self):
        return self.h + self.g

    def print_vert(self):
        print("j1 {}, j2 {}".format(self.j1, self.j2))

class Edge:
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.dis = self.distance()

    def distance(self):
        """
        manhattan distance
        :return:
        """
        return abs(self.v1.j1-self.v2.j1) + abs(self.v1.j2 - self.v2.j2)


class AStar:
    def __init__(self, graph, start_node, end_node):
        self.open_list = []
        self.close_list = []
        self.graph = graph
        self.start_node = start_node
        self.end_node = end_node
        self.current_node = start_node
        self.path_list = []


    def get_min_f_node(self):
        """
        Get the node from openlist with minimum F value
        """
        node_temp = self.open_list[0]
        for node in self.open_list:
            if node.getF() < node_temp.getF():
                node_temp = node
        return node_temp

    def check_identical(self, n1, n2):
        if n1.j1 == n2.j1 and n1.j2 == n2.j2:
            return True
        return False

    def node_in_openlist(self, node):
        if node in self.open_list:
            return True
        return False

    def node_in_closelist(self, node):
        if node in self.close_list:
            return True
        return False

    def make_path(self, node):
        while node:
            self.path_list.append(node)
            node = node.father
        self.path_list.reverse()

    def heuristic_function(self, node):
        e = Edge(node, self.end_node)
        return e.distance()

    def find_path(self):
        """
        Main Calculation
        """
        print("starting planning path from :")
        self.start_node.print_vert()
        print("to :")
        self.end_node.print_vert()
        print("----------------")
        self.start_node.setG(0)
        self.start_node.setH(self.heuristic_function(self.start_node))
        self.open_list.append(self.start_node)
        while len(self.open_list) > 0:
            node = self.get_min_f_node()
            if self.check_identical(node, self.end_node):
                self.make_path(node)
                return True

            self.close_list.append(node)
            self.open_list.remove(node)
            neighbours = self.graph.get_neighbours(node)
            n_list = []
            for n in neighbours:
                if self.node_in_closelist(n):
                    continue
                if self.node_in_openlist(n):
                    if n.g > node.g:
                        n.setFather(node)
                        n.g = node.g + Edge(node, n).distance()
                else:
                    n.setFather(node)
                    n.g = node.g + Edge(node, n).distance()
                    n_list.append(n)
            self.open_list += n_list
        print("Can't find a path.")
        return False

    def get_path(self):
        return self.path_list

    def get_close_list(self):
        return self.close_list

    def get_open_list(self):
        return self.open_list


class Graph:
    def __init__(self, vertexes, edges):
        self.v_list = vertexes
        self.e_list = edges

    def get_neighbours(self, v):
        if v not in self.v_list:
            print("v is not in graph")
            return None
        neighours = []
        for e in self.e_list:
            if e.v1 == v:
                neighours.append(e.v2)
            if e.v2 == v:
                neighours.append(e.v1)
        neighours = list(dict.fromkeys(neighours))
        return neighours


class PRM:
    """
    Probabilistc Road Map
    Randomly generate vertexes in configuration space
    Connect them into a graph
    Plan a path on the graph using A* algorithm
    """
    def __init__(self, scene=Scene(), num_nodes=int(1e3)):
        self.scene = scene
        self.num_nodes = num_nodes  # num of vertexes in the graph
        self.verts = []
        self.edges = []
        self.k = 5  # check nearest 5 neighbours
        self.d_mat = np.zeros((self.num_nodes + 2, self.num_nodes + 2))
        self.graph = self.generate_graph()
        self.path = []
        self.path_edges = []
        self.open_list = []
        self.close_list = []

    def generate_graph(self):
        """
        Generate the Graph for PRM
        :return:
        """
        # generate vertexes
        for i in range(self.num_nodes):
            j1, j2 = self.scene.random_virtual_valid_pose()
            v = Vertex(j1, j2)
            self.verts.append(v)

        # generate distance matrix
        d_mat = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i, self.num_nodes):
                e = Edge(self.verts[i], self.verts[j])
                d_mat[i, j] = e.distance()
        self.d_mat[0:self.num_nodes, 0:self.num_nodes] = d_mat
        d_mat += d_mat.transpose()

        # Connect the edges of a vertex with its k nearest neighbours
        for i in range(self.num_nodes):
            arr = d_mat[i]
            ind = np.argpartition(arr, self.k + 1)[:self.k + 1]
            for j in range(1, self.k + 1):
                e = Edge(self.verts[i], self.verts[ind[j]])
                if not self.edge_in_collision(e):
                    self.edges.append(e)
        return Graph(self.verts, self.edges)

    def edge_in_collision(self, e):
        """
        Using NN to determine whether an edge is in collision with obstacles
        :return:
        """
        if random.random() > 0.0:
            return False  # No collision
        return True

    def connect_neighouring_node(self, p, start_point=True):
        """
        Adding the starting and goal point into the graph
        :param p: a point in configuration space
        :return:
        """
        neighbours = 10
        d_arr = np.zeros((len(self.verts)))
        for index, v in enumerate(self.verts):
            e = Edge(p, v)
            d_arr[index] = e.distance()
        #if start_point:
        #    self.d_mat[self.num_nodes, :] = d_arr
        #else:
        #    self.d_mat[self.num_nodes + 1, :] = d_arr  # Lower triangular matrix
        ind = np.argsort(d_arr)[:neighbours]
        for i in range(len(ind)):
            v = self.verts[ind[i]]
            e = Edge(p, v)
            if not self.edge_in_collision(e):
                self.edges.append(e)
                self.verts.append(p)
                #if start_point:
                #    print("start point is connected to :")
                #    v.print_vert()
                #else:
                #    print("end point is connected to :")
                #    v.print_vert()
                #if not start_point:
                #    self.d_mat += self.d_mat.transpose()  # symmetric matrix
                return
        print("Error: Can't insert the node into the graph.")
        return

    def generate_path_edge(self):
        path_edges = []
        for i in range(len(self.path) - 1):
            e = Edge(self.path[i], self.path[i+1])
            path_edges.append(e)
        return path_edges


    def plan(self, start_point, goal_point):
        self.start_point = start_point
        self.goal = goal_point
        self.connect_neighouring_node(start_point, True)
        self.connect_neighouring_node(goal_point, False)
        astar = AStar(self.graph, self.start_point, self.goal)
        if astar.find_path():
            self.path = astar.get_path()
            self.path_edges = self.generate_path_edge()
            self.open_list = astar.get_open_list()
            self.close_list = astar.get_close_list()
            return self.path
        return None

    def visualize(self):
        area = 0.01
        x_list = []
        y_list = []
        for v in self.verts:
            x_list.append(v.j1)
            y_list.append(v.j2)
        plt.figure()
        ax = plt.subplot(111)
        plt.scatter(x_list, y_list, s=area)
        for e in self.edges:
           ax.plot(
               [e.v1.j1, e.v2.j1],
               [e.v1.j2, e.v2.j2],
               'og-')
        for p in self.close_list:
            ax.plot(
                [p.j1],
                [p.j2],
                'oy'
            )
        for p in self.open_list:
            ax.plot(
                [p.j1],
                [p.j2],
                'ob'
            )
        for e in self.path_edges:
            ax.plot(
                [e.v1.j1, e.v2.j1],
                [e.v1.j2, e.v2.j2],
                'or-'
            )
        plt.show()


if __name__ == "__main__":
    obs_rate = 0.1
    random_obstacle = False
    vis = False
    occ = OccupancyGrid(random_obstacle=random_obstacle, obstacle_probability=obs_rate)
    scene = Scene(visualize=vis, env=occ)
    st = time.time()
    prm = PRM(scene=scene, num_nodes=1000)
    et = time.time()
    print("init time consumption: ", et - st)
    start_point = Vertex(1.0, 1.0)
    goal_point = Vertex(5.0, 3.0)
    st = time.time()
    prm.plan(start_point, goal_point)
    et = time.time()
    print("time consumption: ", et - st)
    prm.visualize()

    # for n in range(10):
    #     v_list = []
    #     for i in range(2):
    #         j1, j2 = scene.random_virtual_valid_pose()
    #         v_list.append(Vertex(j1, j2))
    #     start_point = v_list[0]
    #     goal_point = v_list[1]
    #     st = time.time()
    #     prm.plan(start_point, goal_point)
    #     et = time.time()
    #     print("time consumption: ", et - st)
    #prm.visualize()