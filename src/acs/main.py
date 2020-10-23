import math
import time
from graph import Graph, City
from plot import plot
from acs import ACS

def main():
    import pathlib
    print(pathlib.Path(__file__).parent.absolute())
    print(pathlib.Path().absolute())
    print('Start!')
    cities = []
    points = []
    print('Reading graph')
    with open('input/berlin52.txt') as f:
        for line in f.readlines():
            city = line.split(' ')
            cities.append(City(int(city[1]), int(city[2])))
            points.append((int(city[1]), int(city[2])))
    cost_matrix = []
    rank = len(cities)
    for i in range(rank):
        row = []
        for j in range(rank):
            row.append(cities[i].distance(cities[j]))
        cost_matrix.append(row)

    aco = ACS(n=1000, m=10, alpha=1, beta=5, rho=0.1, phi=0.1, q_zero=0.9)
    graph = Graph(cost_matrix, rank)
    print('Solving TSP-ACS')
    start_time = time.time()
    path, cost = aco.solve(graph)
    print("--- %s seconds ---" % (time.time() - start_time))
    print('Final cost: {}'.format(cost))
    print('Path: {}'.format(path))
    plot(points, path)

if __name__ == '__main__':
    main()