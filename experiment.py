import string
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from genetic_algorithm import TreeGeneticAlgorithm, Tournament
from lib.arboles import SubNode, AddNode, MultNode, MaxNode


class Experiment:
    def __init__(self, population_sizes, mutation_rates, target, genetic_algorithm_class, selection_function,
                 allowed_functions, allowed_parameters, consider_depth=False, allow_repetitions=True):
        self.allowed_parameters = allowed_parameters
        self.allowed_functions = allowed_functions
        self.population_sizes = population_sizes
        self.mutation_rates = mutation_rates
        self.target = target
        self.averages = []
        self.maxes = []
        self.mins = []
        self.results = np.full((len(population_sizes), len(mutation_rates)), np.inf)
        self.genetic_algorithm_class = genetic_algorithm_class
        self.selection_function = selection_function
        self.consider_depth = consider_depth
        self.allow_repetitions = allow_repetitions

    def run_experiment(self, create_heatmap=True):
        for i, population_size in enumerate(self.population_sizes):
            print('population_size: ' + str(population_size))
            for j, mutation_rate in enumerate(self.mutation_rates):
                print('mutation_rate: ' + str(mutation_rate))
                genetic_algorithm_instance = self.genetic_algorithm_class(
                    self.target,
                    population_size,
                    mutation_rate,
                    self.selection_function,
                    self.allowed_functions,
                    self.allowed_parameters,
                )
                genetic_algorithm_instance.evaluate_fitness(self.allow_repetitions, self.consider_depth)
                self.run_iterations(genetic_algorithm_instance, i, j, 100, create_heatmap)

    def run_iterations(self, genetic_algorithm_instance, i, j, max_iterations=100, create_heatmap=True):
        iterations = 1
        while True:
            if not create_heatmap:
                fitness_list = genetic_algorithm_instance.fitness_list
                fitness_list = [f for f in fitness_list if f < self.target]
                fitness_list = [self.target] if len(fitness_list) == 0 else fitness_list
                self.maxes += [max(fitness_list)]
                self.mins += [min(fitness_list)]
                self.averages += [sum(fitness_list) / len(fitness_list)]
            genetic_algorithm_instance.select()
            genetic_algorithm_instance.reproduce()
            genetic_algorithm_instance.evaluate_fitness(self.allow_repetitions, self.consider_depth)
            iterations += 1
            if iterations % 100 == 0:
                print('a')
            if iterations >= max_iterations:
                break
        if not create_heatmap:
            fitness_list = genetic_algorithm_instance.fitness_list
            fitness_list = [f for f in fitness_list if f < self.target]
            fitness_list = [self.target] if len(fitness_list) == 0 else fitness_list
            self.maxes += [max(fitness_list)]
            self.mins += [min(fitness_list)]
            self.averages += [sum(fitness_list) / len(fitness_list)]
        if create_heatmap:
            self.results[i][j] = iterations

    def graph(self, heatmap=True):
        if heatmap:
            self.graph_heatmap()
        else:
            self.graph_evolution()

    def graph_heatmap(self):
        fig, ax = plt.subplots()
        heatmap = ax.pcolor(self.results, norm=colors.LogNorm(vmin=self.results.min(), vmax=self.results.max()))
        column_labels = self.population_sizes
        row_labels = self.mutation_rates
        plt.locator_params(axis='y', nbins=len(column_labels))
        plt.locator_params(axis='x', nbins=len(row_labels))
        ax.set_xticklabels(row_labels, minor=False)
        ax.set_yticklabels(column_labels, minor=False)
        ax.set_xticks([float(n) + 0.5 for n in ax.get_xticks()])
        ax.set_yticks([float(n) + 0.5 for n in ax.get_yticks()])
        ax.set(xlim=(0, len(row_labels)), ylim=(0, len(column_labels)))
        plt.title('Número de iteraciones según cantidad de mutaciones\ny tamaño de la población')
        plt.xlabel('Ratio de mutaciones')
        plt.ylabel('Tamaño de la población')
        plt.colorbar(heatmap)
        plt.savefig('img/heatmap.png')
        plt.show()

    def graph_evolution(self):
        #plt.plot(range(0, len(self.maxes)), self.maxes, label='Máximo por iteración')
        plt.plot(range(0, len(self.mins)), self.mins, label='Mínimo por iteración')
        plt.plot(range(0, len(self.averages)), self.averages, label='Promedio por iteración')
        plt.xlabel('Iteraciones')
        plt.ylabel('Fitness')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crea un algoritmo genético y lo entrena para resolver un problema')
    parser.add_argument(
        'problem',
        type=str,
        default='binary',
        help='Tipo de problema a resolver, puede ser binary, alphanumeric o maze'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-heatmap', action='store_true')
    group.add_argument('-evolution_graph', action='store_true')
    args = parser.parse_args()
    population_sizes = []
    mutation_rates = []
    binary_word = ''
    alphabet_word = ''
    maze = None
    if args.heatmap:
        population_sizes = range(50, 1000, 50)
        mutation_rates = [float(x/100.0) for x in range(0, 10, 1)]
    elif args.evolution_graph:
        population_sizes = [1000]
        mutation_rates = [0.1]
    if args.problem == 'consider_depth':
        experiment = Experiment(population_sizes, mutation_rates, 65346, TreeGeneticAlgorithm, Tournament(5),
                                [SubNode, AddNode, MaxNode, MultNode], [25, 7, 8, 100, 4, 2], False, True)
        experiment.run_experiment(args.heatmap)
        experiment.graph(args.heatmap)
    elif args.problem == 'consider_repetitions':
        experiment = Experiment(population_sizes, mutation_rates, 65346, TreeGeneticAlgorithm, Tournament(5),
                                [SubNode, AddNode, MaxNode, MultNode], [25, 7, 8, 100, 4, 2], False, False)
        experiment.run_experiment(args.heatmap)
        experiment.graph(args.heatmap)
    elif args.problem == 'maze':
        _, solution = maze.get_shortest_path()
        experiment = Experiment(population_sizes, mutation_rates, solution, TreeGeneticAlgorithm,
                                Tournament(5))
        experiment.run_experiment(args.heatmap)
    print('done')
