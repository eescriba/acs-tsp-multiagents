import math
import time

from .ga import City, Fitness, geneticAlgorithmPlot

def main():
    print('Start!')
    cities = []
    points = []
   
    print('Reading graph')
    with open('input/att48.txt') as f:
        for line in f.readlines():
            city = line.split(' ')
            cities.append(City(int(city[1]), int(city[2])))
            points.append((int(city[1]), int(city[2])))

    print('Solving TSP-GA')
    start_time = time.time()
    path = geneticAlgorithmPlot(population=cities, popSize=100, eliteSize=20, mutationRate=0.001, generations=500)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(path)

if __name__ == '__main__':
    main()