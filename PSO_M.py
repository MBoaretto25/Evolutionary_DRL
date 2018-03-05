# -*- coding: utf-8 -*-
"""
PSO for Deep Reinforced Learning
based on pso from pyswarm package

@author: marco
"""
## imports 
from pso_func import pso_func
from dqn import dqn
import time

# Get initial time for runs
start_time = time.time()

## Define fitness function
def fitness(x):
    
#    fit_val = sum(x)
    fit_val = dqn(x)

    return -fit_val  

## Define constraint 
def con(x):
    x0 = x[0] 
    x1 = x[1]
    x2 = x[2]
    x3 = x[3]
    x4 = x[4]     
    return [1-x0,128-x1,3-x2,3-x3,1-x4]

## Define the Lower and Upper limits f_ieqcons=con
lb = [0, 8  , 0 ,0 ,.10]
ub = [2, 129, 4 ,4 ,.99]

## Start optimization
xopt, fopt, fit_hist = pso_func(fitness, lb, ub, f_ieqcons=con,swarmsize=10,maxiter=50,debug=True)

## Present results
nm = time.time() - start_time
print("Best Fitness fun = {:} from X[{:}] in {:} s".format(fopt,xopt,nm))
