import random
import numpy as np
from acs.acs import ACS, Ant
from acs.graph import Graph

from .hmm import HMM

# Ant Colony System with HMM parameter opt
class ACS_HMM(ACS):

    hmm = HMM()
    maxiter = 10
    maxdist = 0

    def solve(self, graph: Graph):

        best_cost = float("inf")
        best_solution = []
        best_ant_pos = -1
        self.maxdist = max(max(graph.matrix))

        for gen in range(self.n):
            self.hmm = HMM()
            print("Generation", gen)
            for it in range(self.maxiter):
                ants = [Ant(self, graph) for i in range(self.m)]
                for ant in ants:
                    ant.generate_tour()
                    if ant.total_cost < best_cost:
                        best_cost = ant.total_cost
                        print("Best cost update: ", best_cost)
                        best_solution = [] + ant.tour
                        best_ant_pos = ant.current
                new_phi, new_rho = self.update_pheromone_params(graph, ants, best_ant_pos, best_cost, it)
                if self.phi != new_phi or self.rho != new_rho:
                    print("Update rho/phi: ", '{0:.2f}'.format(new_rho), "/", '{0:.2f}'.format(new_phi))
                    self.phi = new_phi
                    self.rho = new_rho
            self.update_pheromone(graph, ants, best_solution, best_cost)
       
        return best_solution, best_cost

     # HMM update rho/phi
    def update_pheromone_params(self, graph: Graph, ants: list, best_ant_pos: int, best_cost: float, current_i: int):
        iteration = current_i/self.maxiter
        diversity = self.calc_diversity(graph, ants, best_ant_pos)/self.maxdist
        return self.hmm.update_params(iteration, diversity, self.rho, self.phi)

    # Ant dispersion
    def calc_diversity(self, graph: Graph, ants: list, best_ant_pos: int):
        return (1/len(ants)) * sum(graph.matrix[best_ant_pos][ant.current] for ant in ants)