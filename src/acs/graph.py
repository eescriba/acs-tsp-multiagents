import numpy as np
import random

class Graph(object):
    def __init__(self, cost_matrix: list, N: int):
        """
        :param cost_matrix: cost matrix
        :param N: number of nodes
        :param pheromone: global pheromone
        """
        self.matrix = cost_matrix
        self.N = N
        cnn = self.nn_heuristic(cost_matrix, N)
        self.initial_pheromone = [[1 / (N * cnn) for j in range(N)] for i in range(N)]
        self.pheromone = self.initial_pheromone
        # heuristic value
        self.eta = [[0 if i == j else 1 / cost_matrix[i][j] for j in range(N)] for i in range(N)] 

    def nn_heuristic(self, cost_matrix: list, N: int):
        total_cost = 0
        unvisited = list(range(N))
        start = random.randint(0, N - 1)
        unvisited.remove(start)
        selected = start
    
        while unvisited:
            mincost = np.inf
            nearest = -1
            for i in unvisited:
                cost = cost_matrix[selected][i]
                if cost < mincost:
                    mincost = cost
                    nearest = i
            total_cost += mincost
            selected = nearest
            unvisited.remove(selected)
        total_cost+=cost_matrix[selected][start]
        return total_cost

class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance
    
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"
