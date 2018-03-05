import numpy as np
import deap
from deap import algorithms, base, tools
import imp
from cachetools import LRUCache

# used to control the size of the cache so that it doesn't exceed system memory
MAX_MEM_BYTES = 11E9


class GeneticDetMinimizer(object):

    def __init__(self, N=30, popsize=500, cachesize=None, seed=0):

        # an 'individual' consists of an (N^2,) flat numpy array of 0s and 1s
        self.N = N
        self.indiv_size = N * N

        if cachesize is None:
            cachesize = int(np.ceil(104 * MAX_MEM_BYTES / 10))

        self._gen = np.random.RandomState(seed)

        # we want the creator module to be local to this instance, since
        # creator.create() directly adds new classes to the module's globals()
        # (yuck!)
        cr = imp.load_module('cr', *imp.find_module('creator', deap.__path__))
        self._cr = cr

        self._cr.create("FitnessMin", base.Fitness, weights=(-1.0,))
        self._cr.create("Individual", np.ndarray, fitness=self._cr.FitnessMin)

        self._tb = base.Toolbox()
        self._tb.register("attr_bool", self.random_bool)
        self._tb.register("individual", tools.initRepeat, self._cr.Individual,
                          self._tb.attr_bool, n=self.indiv_size)

        # the 'population' consists of a list of such individuals
        self._tb.register("population", tools.initRepeat, list,
                          self._tb.individual)
        self._tb.register("evaluate", self.fitness)
        self._tb.register("mate", self.crossover)
        self._tb.register("mutate", self.mutate, rate=0.002)
        self._tb.register("select", tools.selTournament, tournsize=3)

        # create an initial population, and initialize a hall-of-fame to store
        # the best individual
        self.pop = self._tb.population(n=popsize)
        self.hof = tools.HallOfFame(1, similar=np.array_equal)

        # print summary statistics for the population on each iteration
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

        # keep track of configurations that have already been visited
        self.tabu = LRUCache(cachesize)

    def random_bool(self, *args):
        return self._gen.rand(*args) < 0.5

    def mutate(self, ind, rate=1E-3):
        """
        mutate an individual by bit-flipping one or more randomly chosen
        elements
        """
        # ensure that each mutation always introduces a novel configuration
        while np.packbits(ind.astype(np.uint8)).tostring() in self.tabu:
            n_flip = self._gen.binomial(self.indiv_size, rate)
            if not n_flip:
                continue
            idx = self._gen.random_integers(0, self.indiv_size - 1, n_flip)
            ind[idx] = ~ind[idx]
        return ind,

    def fitness(self, individual):
        """
        assigns a fitness value to each individual, based on the determinant
        """
        print("Individual = {}".format(individual))
        h = np.packbits(individual.astype(np.uint8)).tostring()
        # look up the fitness for this configuration if it has already been
        # encountered
        print("H = {}".format(h))

        if h not in self.tabu:
            fitness = np.linalg.det(individual.reshape(self.N, self.N))
            self.tabu.update({h: fitness})
        else:
            fitness = self.tabu[h]

        print("fitness = {}".format(fitness))
        return fitness,

    def crossover(self, ind1, ind2):
        """
        randomly swaps a block of rows or columns between two individuals
        """

        cx1 = self._gen.random_integers(0, self.N - 2)
        cx2 = self._gen.random_integers(cx1, self.N - 1)
        ind1.shape = ind2.shape = self.N, self.N

        if self._gen.rand() < 0.5:
            # row swap
            ind1[cx1:cx2, :], ind2[cx1:cx2, :] = (
                ind2[cx1:cx2, :].copy(), ind1[cx1:cx2, :].copy())
        else:
            # column swap
            ind1[:, cx1:cx2], ind2[:, cx1:cx2] = (
                ind2[:, cx1:cx2].copy(), ind1[:, cx1:cx2].copy())

        ind1.shape = ind2.shape = self.indiv_size,

        return ind1, ind2

    def run(self, ngen=int(5), mutation_rate=0.3, crossover_rate=0.7):

        pop, log = algorithms.eaSimple(self.pop, self._tb,
                                       cxpb=crossover_rate,
                                       mutpb=mutation_rate,
                                       ngen=ngen,
                                       stats=self.stats,
                                       halloffame=self.hof)
        self.log = log

        return self.hof[0].reshape(self.N, self.N), log

if __name__ == "__main__":
    np.random.seed(0)
    gd = GeneticDetMinimizer(10)
    best, log = gd.run()