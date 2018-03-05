# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 09:03:39 2018

Code adapted from : https://stackoverflow.com/users/1461210/ali-m

@article{DEAP_JMLR2012, 
    author    = " F\'elix-Antoine Fortin and Fran\c{c}ois-Michel {De Rainville} and Marc-Andr\'e Gardner and Marc Parizeau and Christian Gagn\'e ",
    title     = { {DEAP}: Evolutionary Algorithms Made Easy },
    pages    = { 2171--2175 },
    volume    = { 13 },
    month     = { jul },
    year      = { 2012 },
    journal   = { Journal of Machine Learning Research }
}

@author = Marco Boaretto

"""

import numpy as np
import deap
from deap import algorithms, base, tools
import imp
import random
import matplotlib.pyplot as plt
from dqn import dqn
import time
import pickle  
#import gym
from cachetools import LRUCache

## Load Problem Here for code efficiency
start_time = time.time();


# used to control the size of the cache so that it doesn't exceed system memory
MAX_MEM_BYTES = 11E9

class GeneticDetMinimizer(object):

    def __init__(self, popsize=10, cachesize=None, seed=234):
       
        self._gen = np.random.RandomState(seed)     
#        self._FEVAL = 0;
        
        if cachesize is None:
            cachesize = 10e6
#            cachesize = int(np.ceil(144 * MAX_MEM_BYTES / 10))
        
        self._popsize = popsize
        
        # we want the creator module to be local to this instance, since
        # creator.create() directly adds new classes to the module's globals()
        cr = imp.load_module('cr', *imp.find_module('creator', deap.__path__))
        self._cr = cr
        
        ## Creator for a Maximization Fun, weiths = 1
        self._cr.create("FitnessMax", base.Fitness, weights=(1.0,))       
        ## Creator for a Individual
        self._cr.create("Individual", list , fitness=self._cr.FitnessMax)      

        ## Creating the Variables to be tuned
        self._tb = base.Toolbox()        
        # layers size
        self._tb.register("layer_size", random.randint, 0, 1)
        # num_units
        self._tb.register("num_units", random.randint, 8, 128)
        # activ_fun
        self._tb.register("activ_fun", random.randint, 0, 3)
        # optimizer
        self._tb.register("optimizer", random.randint, 0, 3)
        # w_dropout
        self._tb.register("w_dropout", self.round_random,2)
        
        # Structure initializers
        self._tb.register("individual", tools.initCycle, self._cr.Individual,
                 (self._tb.layer_size, 
                  self._tb.num_units,
                  self._tb.activ_fun,
                  self._tb.optimizer,
                  self._tb.w_dropout))
        
        # the 'population' consists of a list of such individuals
        self._tb.register("population", tools.initRepeat, list,
                          self._tb.individual)
        self._tb.register("evaluate", self.fitness)
        self._tb.register("mate", tools.cxUniform,indpb = 0.6)
        self._tb.register("mutate", self.mutate)
        self._tb.register("select", tools.selTournament, tournsize=3)

        # create an initial population, and initialize a hall-of-fame to store
        # the best individual
        self.pop = self._tb.population(n=10)
        self.hof = tools.HallOfFame(1)

        # print summary statistics for the population on each iteration
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        
        # keep track of configurations that have already been visited
        self.tabu = LRUCache(cachesize)
        
    def round_random(self,x):    
        return round(random.random(),1) 
    
    def mutate(self, ind1):
        """
        # ensure that each mutation always introduces a novel configuration        
        """        
        mut = self._tb.clone(ind1);
        
        mut[0] = random.randint(0, 1)
        mut[1] = random.randint(8, 128)
        mut[2],mut[3] = [random.randint(0, 3),random.randint(0, 3)]
        mut[4] = random.random()
        
        ind2 = mut
        
        del mut.fitness.values
        
        return ind2,

    def fitness(self, individual):
        """
        assigns a fitness value to each individual, based on the determinant
        """
        h = str(individual)
        # look up the fitness for this configuration if it has already been
        # encountered
        if h not in self.tabu:
            fitness = dqn(individual)
            self.tabu.update({h: fitness})
        else:
            fitness = self.tabu[h]  
        
#        self._FEVAL += 1
        
#        print("Feval {} and current cache = {} of {} ".format(self._FEVAL,self.tabu.currsize,self.tabu.maxsize))
        
        return fitness,        
    
    def run(self, ngen, mutation_rate=0.5, crossover_rate=0.8):

        pop, log = algorithms.eaSimple(self.pop, self._tb,
                                       cxpb=crossover_rate,
                                       mutpb=mutation_rate,
                                       ngen=ngen,
                                       stats=self.stats,
                                       halloffame=self.hof)
        
        self.log = log
        
        return self.hof[0], log

if __name__ == "__main__":
    ## Start Optimization
    
    max_gen = 1
    np.random.seed(0)
    gd = GeneticDetMinimizer(0)
    best, log = gd.run(ngen = max_gen)    
    
    ## End Optimization

    
    ## Results        
    gen = log.select("gen")
    fit_max = log.select("max")        
    fit_avg= log.select("avg")   
    
    end_time = time.time()-start_time;
    print("\nTotal Run Time with {} iters in {} seconds, Best val {}\n".format(max_gen,end_time,np.max(fit_max)))
    
    # Saving the objects:    
    with open('LogRes_GA_.pkl', 'wb') as f:  
        pickle.dump([log, best, end_time], f)
        
    # Ploting results
    plt.plot(gen, fit_max, "b-", label="Best Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness", color="b")
    plt.title('Convergence Curve')  
    
    plt.show()

    # Getting back the objects:
    # to load 
    # 1) define the functions again 
    # 2) gd = GeneticDetMinimizer(0)
    with open('LogRes_GA_.pkl','rb') as f:  
        log1, best1, end_time1 = pickle.load(f)
    

