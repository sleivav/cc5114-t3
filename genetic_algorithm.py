import random

import numpy as np
from abc import abstractmethod, ABC

from lib.arboles import *

np.random.seed(5114)
random.seed(5144)


class SelectionFunction(ABC):
    @abstractmethod
    def select(self, fitness_list, population):
        pass


class Roulette(SelectionFunction):
    def select(self, fitness_list, population) -> list:
        probabilities = self.generate_probabilities(fitness_list)
        choices = np.random.choice(len(population), 2*len(population), p=probabilities)
        return [population[i] for i in choices]

    @staticmethod
    def generate_probabilities(fitness_list) -> list:
        total = sum(fitness_list)
        return [float(fitness_list[i]) / float(total) for i in range(len(fitness_list))]


class Tournament(SelectionFunction):
    def __init__(self, tournament_size=5):
        self.tournament_size = tournament_size

    def select(self, fitness_list, population) -> list:
        best_individual = population[fitness_list.index(max(fitness_list))]
        selection = [best_individual]
        for i in range(2*len(population) - 1):
            selection_indices = np.random.choice(len(population), self.tournament_size)
            fitnesses = [fitness_list[i] for i in selection_indices]
            max_fit_index = selection_indices[fitnesses.index(max(fitnesses))]
            selection += [population[max_fit_index]]
        return selection


class TreeGeneticAlgorithm:
    def __init__(self, target_result: str, population_size: int, mutation_rate: float, selection_function: SelectionFunction,
                 allowed_functions: list, allowed_parameters: list):
        self.target_result = target_result
        self.allowed_functions = allowed_functions
        self.allowed_parameters = allowed_parameters
        self.population = [self.generate_individual() for _ in range(population_size)]
        self.fitness_list = []
        self.selection_function = selection_function
        self.selection = []
        self.mutation_rate = mutation_rate

    def evaluate_fitness(self) -> None:
        fitness = []
        for individual in self.population:
            fitness += [self.get_ind_fitness(individual)]
        self.fitness_list = fitness

    '''
    Utiliza la selección de función para elegir los individuos que serán padres de la próxima generación
    '''
    def select(self):
        self.selection = self.selection_function.select(self.fitness_list, self.population)

    '''
    Utiliza los individuos seleccionados para generar la nueva generación
    '''
    def reproduce(self):
        for i in range(0, len(self.selection), 2):
            self.population[i // 2] = self.mutate(self.crossover(self.selection[i], self.selection[i+1]))

    '''
    Genera un nuevo individuo a partir de otros dos
    '''
    def crossover(self, ind1: Node, ind2: Node):
        new_element = ind1.copy()
        p1 = random.choice(new_element.serialize())
        p2 = random.choice(ind2.serialize()).copy()
        p1.replace(p2)
        return new_element

    '''
    Muta un individuo, cambio uno de sus genes por otro, al azar
    '''
    def mutate(self, ind):
        random_node = random.choice(ind.serialize())
        new_node = self.generate_individual(random.randrange(11))
        random_node.replace(new_node)
        return ind

    def get_ind_fitness(self, individual):
        return abs(individual.eval() - self.target_result)

    def generate_individual(self, max_depth=10):
        ast = AST(self.allowed_functions, self.allowed_parameters)
        return ast(max_depth)
