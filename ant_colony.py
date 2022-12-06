import sys, string, os
import subprocess
from pyswarm import pso
import pants
import math
import random
import numpy as np

# number of parameters
NPARAMS = 2
# discretized parameter 1: threads
threads = [1, 2, 3, 4, 5]
# discretized parameter 2: cache blocking
caches = [2, 4, 8, 16]

NTHREADS = len(threads)
NCACHE = len(caches)
# maximum number of options
MAXPARAM = 5
# number of ants
NANTS = 5
# evaporation rate
EVAP_RATE = 0.5
# number of iterations
MAXITERS = 10

# pheromone matrix
pheromone = np.zeros((NPARAMS,NANTS))
# probability matrix
probabilities = np.zeros((NPARAMS,NANTS))
# cost fucntion arrary
costs = np.zeros(NANTS, dtype=int)

selected_thread = np.zeros(NANTS, dtype=int)
#int     selected_cache[NANTS];
selected_cache = np.zeros(NANTS, dtype=int)


def cost(threads, cache):
    return(threads**2 + cache**2)

def initialize():

    print("Initializing problem...")
    print("")
    
    i = 0
    j = 0

    print("Thread values: ")
    print(threads)

    print("")
    
    print("Cache values: ")
    print(caches)

    print("")

    for i in range(NPARAMS):
        for j in range(MAXPARAM):
            pheromone[i,j] = 0

    for i in range(NTHREADS):
        pheromone[0,i] = 1.0

    for i in range(NCACHE):
        pheromone[1,i] = 1.0

    for i in range(NTHREADS):
        probabilities[0,i] = 1.0/(float(NTHREADS))

    for i in range(NCACHE):
        probabilities[1,i] = 1.0/(float(NCACHE))

def ants_pick_paths():

    global lowest_cost
    global highest_cost
    global best_ant
    global worst_ant

    lowest_cost = 1000000 #infinity
    highest_cost = 0
    
    for i in range(NANTS):
        cumulative_prob = 0
        roulette  = round(random.random(),2)

        for j in range(NTHREADS):
            if roulette <= (cumulative_prob + probabilities[0,j]):
                selected_thread[i] = j
                break
            else:
                cumulative_prob += probabilities[0,j]

        cumulative_prob = 0
        
        roulette = round(random.random(),2)
        
        for j in range(NCACHE):
            if roulette <= (cumulative_prob + probabilities[1,j]):
                selected_cache[i] = j
                break
            else:
                cumulative_prob += probabilities[1,j]
        
        costs[i] = cost(threads[selected_thread[i]], caches[selected_cache[i]])
        
        if costs[i] < lowest_cost:
            lowest_cost = costs[i]
            best_ant = i

        if costs[i] > highest_cost:
            highest_cost = costs[i]
            worst_ant = i

def update_pheromone():

    for i in range(NPARAMS):
        for j in range(NANTS):
            if j == best_ant:
                pheromone[i,j] += 2*(lowest_cost/highest_cost)
            else:
                pheromone[i,j] = (1 - EVAP_RATE)*pheromone[i,j]

def update_probabilities():

    sum_pheromones = np.zeros(NPARAMS)

    for i in range(NPARAMS):
        for j in range(MAXPARAM):
            sum_pheromones[i] += pheromone[i,j]

    for i in range(NPARAMS):
        for j in range(MAXPARAM):
            probabilities[i,j] = (pheromone[i,j])/(sum_pheromones[i])

def display_solution():

    print("Pheromones: ")
    print(pheromone)

    print("")

    print("Probabilities: ")
    print(probabilities)

    print("")

    for i in range(NANTS):
        print("Ant number: " + str(i))
        print("Thread[" + str(selected_thread[i]) + "]: " + str(threads[selected_thread[i]]))
        print("Cache[" + str(selected_cache[i]) + "]: " + str(caches[selected_cache[i]]))
        print("Cost: " + str(cost(threads[selected_thread[i]],caches[selected_cache[i]])))

    print("")
    print("Lowest cost: " + str(lowest_cost))
    print("Best ant: " + str(best_ant))
    print("Highest cost: " + str(highest_cost))
    print("Worst ant: " + str(worst_ant))
    print("")



if __name__ == "__main__":

    iters = 1

    initialize()

    while (iters < MAXITERS):
        print("ITERATION: " + str(iters))
        print("")
        ants_pick_paths()
        display_solution()
        update_pheromone()
        update_probabilities()
        iters += 1
