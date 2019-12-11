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
        best_individual = population[fitness_list.index(min(fitness_list))]
        selection = [best_individual]
        for i in range(2*len(population) - 1):
            selection_indices = np.random.choice(len(fitness_list), self.tournament_size)
            fitnesses = [fitness_list[i] for i in selection_indices]
            max_fit_index = selection_indices[fitnesses.index(min(fitnesses))]
            selection += [population[max_fit_index]]
        return selection


class TreeGeneticAlgorithm:
    def __init__(self, target_result: int, population_size: int, mutation_rate: float, selection_function: SelectionFunction,
                 allowed_functions: list, allowed_parameters: list, variables: dict = None):
        self.target_result = target_result
        self.allowed_functions = allowed_functions
        self.allowed_parameters = allowed_parameters
        self.population = [self.generate_individual() for _ in range(population_size)]
        self.fitness_list = []
        self.selection_function = selection_function
        self.selection = []
        self.mutation_rate = mutation_rate
        self.differences = []
        self.depths = []
        self.variables = variables

    def evaluate_fitness(self, allow_repetition=True, consider_depth=True, function=None) -> None:
        self.get_differences_and_depths(function)
        fitness = []
        for index, _ in enumerate(self.population):
            fitness += [self.get_ind_fitness(index, allow_repetition, consider_depth)]
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
        a = random.random()
        if a < self.mutation_rate:
            random_node = random.choice(ind.serialize())
            new_node = self.generate_individual(random.randrange(3))
            random_node.replace(new_node)
        return ind

    def get_ind_fitness(self, individual_index, allow_repetition, consider_depth):
        criteria = []
        criteria += [self.differences[individual_index]]
        criteria_number = 1
        if consider_depth:
            criteria += [self.depths[individual_index]] # if self.depths[individual_index] < 15 else 100000]
            criteria_number += 1
        if not allow_repetition:
            has_repeated_terminals = self.ind_has_repeated_terminals(individual_index)
            criteria_number += 2
            if has_repeated_terminals:
                criteria += [self.target_result * 2]
        return sum(criteria) / float(criteria_number)

    def generate_individual(self, max_depth=5):
        ast = AST(self.allowed_functions, self.allowed_parameters)
        return ast(max_depth)

    def get_differences_and_depths(self, function=None):
        self.depths = []
        self.differences = []
        self.depths = [ind.get_depth() for ind in self.population]
        if function is not None:
            self.differences = [sum(abs(ind.eval({'x': x}) - function(x)) for x in range(-100, 101)) for ind in self.population]
        else:
            self.differences = [abs(ind.eval(self.variables) - self.target_result) for ind in self.population]

    def ind_has_repeated_terminals(self, individual_index):
        terminals = [x for x in self.population[individual_index].serialize() if isinstance(x, TerminalNode)]
        return len(set(terminals)) != len(terminals)


if __name__ == '__main__':
    algorithm = TreeGeneticAlgorithm(65346, 1000, 0.05, Tournament(5), [SubNode, AddNode, MaxNode, MultNode],
                                     [25, 7, 8, 100, 4, 2])
    iterations = 0
    algorithm.evaluate_fitness()
    while iterations < 100:
        print(iterations)
        if iterations == 50:
            print('a')
        algorithm.select()
        algorithm.reproduce()
        algorithm.evaluate_fitness()
        iterations += 1
