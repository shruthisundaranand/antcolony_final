import sys, string, os
import subprocess
import math
import random
import numpy as np

# dimentions of paramaters
NTHREAD  = 16
NCACHE1  = 65
NCACHE2  = 12
NCACHE3  = 30

# discretized parameter 1: thread
thread = np.zeros(NTHREAD, dtype=int)

# discretized parameter 2: cache blocking 1
cache1 = np.zeros(NCACHE1, dtype=int)

# discretized parameter 3: cache blocking 2
cache2 = np.zeros(NCACHE2, dtype=int)

# discretized parameter 4: cache blocking 3
cache3 = np.zeros(NCACHE3, dtype=int)

# number of ants
NANTS = 20

# Keep track of solutions computed to avoid recomputation of cost
solutions = np.zeros((NTHREAD,NCACHE1,NCACHE2,NCACHE3), dtype=float)

# True optimal cost = (1*1) + (16*16) + (16*16) + (16*16) = 769 
OPTIMAL_COST = 769

# number of iterations
MAXITERS = 20

# evaporation rate
EVAP_RATE = 0.5

# ALPHA is the constant factor used to increase pheromone
ALPHA = 5

# Algorithm options 
# Evaporation options
# EVAP_OPTION = 0 : evaporate for all paths at a constant rate
# EVAP_OPTION = 1 : evaporate for all paths at constant rate, but no evaporation for best ant
EVAP_OPTION = 0

# Pheromone deposit options
# DEPO_OPTION = 0 : deposit for best ant only
# DEPO_OPTION = 1 : deposit for all ants randomly
# DEPO_OPTION = 2 : deposit for all ants inversely proportional to cost
# DEPO_OPTION = 3 : deposit for all ants and their neighbors as well based on cost and distance
DEPO_OPTION = 3

# pheromone matrix
pheromone = np.zeros((NTHREAD,NCACHE1,NCACHE2,NCACHE3), dtype=float)

# probability vector for thread
probabilities_thread = np.zeros(NTHREAD, dtype=float)

# probability vector for cache1
probabilities_cache1 = np.zeros(NCACHE1, dtype=float)

# probability vector for cache2
probabilities_cache2 = np.zeros(NCACHE2, dtype=float)

# probability vector for cache3
probabilities_cache3 = np.zeros(NCACHE3, dtype=float)

# cost function array for each ant
costs = np.zeros(NANTS, dtype=int)

# tracking exploration of ants & info on best ant & convergence
# the selected indices of parameters 
selected_thread = np.zeros(NANTS, dtype=int)
selected_cache1 = np.zeros(NANTS, dtype=int)
selected_cache2 = np.zeros(NANTS, dtype=int)
selected_cache3 = np.zeros(NANTS, dtype=int)
converged = 0
num_calls = 0

# Other constants
INFINITY = 1000000000

# The objective cost function: sphere function
def cost(threadid, cache1id, cache2id, cache3id):

    global num_calls

    if solutions[threadid, cache1id, cache2id, cache3id] > 0:
        return(solutions[threadid, cache1id, cache2id, cache3id])
    else:
        threadvalue = thread[threadid]
        cache1value = cache1[cache1id]
        cache2value = cache2[cache2id]
        cache3value = cache3[cache3id] 
        result = threadvalue**2 +  cache1value**2 + cache2value**2 + cache3value**2
        solutions[threadid, cache1id, cache2id, cache3id] = result
        num_calls += 1
        return(result)
        
# Initialize parameters and display problem specifications
def initialize():

    print("Initializing problem...")
    print("")
    
    for i in range(NTHREAD):
        thread[i] = i+1
    print("Thread values: ")
    print(thread)
    print("")
    
    for i in range(NCACHE1):
        cache1[i] = 16*(i+1)
    
    print("Cache1 values: ")
    print(cache1)
    print("")

    for i in range(NCACHE2):
        cache2[i] = 16*(i+1)
    print("Cache2 values: ")
    print(cache2)
    print("")

    for i in range(NCACHE3):
        cache3[i] = 16*(i+1)
    print("Cache3 values: ")
    print(cache3)
    print("")

    print("Number of ants: ")
    print(NANTS)
    print("")

    print("Evaporation rate: ")
    print(EVAP_RATE)
    print("")

    print("Objective cost function is the sphere function i.e.")
    print("thread*thread + cache1*cache1 + cache2*cache2 + *cache3*cache3")
    print("")

    for i in range(NTHREAD):
        for j in range(NCACHE1):
            for k in range(NCACHE2):
                for l in range(NCACHE3):          
                    pheromone[i,j,k,l] = 1.0

    for i in range(NTHREAD):
        probabilities_thread[i] = 1.0/(float(NTHREAD))
    for i in range(NCACHE1):
        probabilities_cache1[i] = 1.0/(float(NCACHE1))
    for i in range(NCACHE2):
        probabilities_cache2[i] = 1.0/(float(NCACHE2))
    for i in range(NCACHE3):
        probabilities_cache3[i] = 1.0/(float(NCACHE3))

# Ants pick paths based on roulette wheel method
def ants_pick_paths():

    # these keep information on best and worst ant
    global best_ant
    global worst_ant
    global best_cost  
    global worst_cost 
    global best_thread
    global worst_thread
    global best_cache1
    global worst_cache1
    global best_cache2
    global worst_cache2
    global best_cache3
    global worst_cache3

    best_cost  = INFINITY
    worst_cost = 0
    
    for i in range(NANTS):

	# PARAMETER 1: THREAD 
        # cumulative probability 
        cumulative_prob = 0

	# roulette is uniformly distributed between 0 and 1
        roulette  = random.random()

        for j in range(NTHREAD):
            if roulette <= (cumulative_prob + probabilities_thread[j]):
                selected_thread[i] = j
                break
            else:
                cumulative_prob += probabilities_thread[j]

	# PARAMETER 2: CACHE1
        # cumulative probability 
        cumulative_prob = 0
        
        roulette = random.random()
        
        for j in range(NCACHE1):
            if roulette <= (cumulative_prob + probabilities_cache1[j]):
                selected_cache1[i] = j
                break
            else:
                cumulative_prob += probabilities_cache1[j]

        # PARAMETER 3: CACHE2
        # cumulative probability
        cumulative_prob = 0

        roulette = random.random()

        for j in range(NCACHE2):
            if roulette <= (cumulative_prob + probabilities_cache2[j]):
                selected_cache2[i] = j
                break
            else:
                cumulative_prob += probabilities_cache2[j]

        # PARAMETER 4: CACHE3
        # cumulative probability
        cumulative_prob = 0

        roulette = random.random()

        for j in range(NCACHE3):
            if roulette <= (cumulative_prob + probabilities_cache3[j]):
                selected_cache3[i] = j
                break
            else:
                cumulative_prob += probabilities_cache3[j]
        
        costs[i] = cost(selected_thread[i], \
                        selected_cache1[i], \
                        selected_cache2[i], \
                        selected_cache3[i])

        if costs[i] < best_cost:
            best_cost = costs[i]
            best_ant = i
            best_thread = selected_thread[i]
            best_cache1 = selected_cache1[i]
            best_cache2 = selected_cache2[i]
            best_cache3 = selected_cache3[i]

        if costs[i] > worst_cost:
            worst_cost = costs[i]
            worst_ant = i
            worst_thread = selected_thread[i]
            worst_cache1 = selected_cache1[i]
            worst_cache2 = selected_cache2[i]
            worst_cache3 = selected_cache3[i]

def update_pheromone():

    # evaporate pheromone (for all paths)
    for i in range(NTHREAD):
        for j in range(NCACHE1):
            for k in range(NCACHE2):
                for l in range(NCACHE3):
                    if (EVAP_OPTION == 0): # evaporate for all paths at constant rate
                        pheromone[i,j,k,l] = (1 - EVAP_RATE)*pheromone[i,j,k,l]
                    if (EVAP_OPTION == 1): # no evaporation for best ant
                        if (i==best_thread and j==best_cache1 and k==best_cache2 and l==best_cache3):
                            pheromone[i,j,k,l] = pheromone[i,j,k,l]
                        else:
                            pheromone[i,j,k,l] = (1 - EVAP_RATE)*pheromone[i,j,k,l]

    # deposit pheromone (for ants only)
    if (DEPO_OPTION == 0): # deposit for best ant only
        pheromone[best_thread, \
                  best_cache1, \
                  best_cache2, \
                  best_cache3] += ALPHA*worst_cost/best_cost
    if (DEPO_OPTION == 1): # random deposits for all ants
        for i in range(NANTS):
            pheromone[selected_thread[i], \
                      selected_cache1[i], \
                      selected_cache2[i], \
                      selected_cache3[i]] += ALPHA*random.random()
    if (DEPO_OPTION == 2): # all ants based on cost
        for i in range(NANTS):
            pheromone[selected_thread[i], \
                      selected_cache1[i], \
                      selected_cache2[i], \
                      selected_cache3[i]] += ALPHA*worst_cost/costs[i]
    if (DEPO_OPTION == 3): # all ants and their immediate neighbors based on cost
        for i in range(NANTS):
            pheromone[selected_thread[i], \
                      selected_cache1[i], \
                      selected_cache2[i], \
                      selected_cache3[i]] += ALPHA*worst_cost/costs[i]
        update_neighbors_thread()
        update_neighbors_cache1()
        update_neighbors_cache2()
        update_neighbors_cache3()


def update_neighbors_thread():

    # update two neighbors along thread axis

    # neighbor on the left
    if best_thread > 0:
        neighbor_cost = cost(best_thread-1, \
                             best_cache1,   \
                             best_cache2,   \
                             best_cache3)

        if neighbor_cost < best_cost:
           pheromone[best_thread-1, \
                     best_cache1,   \
                     best_cache2,   \
                     best_cache3] += ALPHA*worst_cost/neighbor_cost

    # neighbor on the right
    if best_thread < NTHREAD-1:
        neighbor_cost = cost(best_thread+1, \
                             best_cache1,   \
                             best_cache2,   \
                             best_cache3)
        if neighbor_cost < best_cost:
            pheromone[best_thread+1, \
                      best_cache1,   \
                      best_cache2,   \
                      best_cache3] += ALPHA*worst_cost/neighbor_cost

def update_neighbors_cache1():

    # update two neighbors along cache1 axis

    # neighbor on the left
    if best_cache1 > 0:
        neighbor_cost = cost(best_thread,   \
                             best_cache1-1, \
                             best_cache2,   \
                             best_cache3) 
        if neighbor_cost < best_cost:
            pheromone[best_thread,   \
                      best_cache1-1, \
                      best_cache2,   \
                      best_cache3] += ALPHA*worst_cost/neighbor_cost

    # neighbor on the right
    if best_cache1 < NCACHE1-1:
        neighbor_cost = cost(best_thread,   \
                             best_cache1+1, \
                             best_cache2,   \
                             best_cache3)
        if neighbor_cost < best_cost:
            pheromone[best_thread,   \
                      best_cache1+1, \
                      best_cache2,   \
                      best_cache3] += ALPHA*worst_cost/neighbor_cost


def update_neighbors_cache2():

    # update two neighbors along cache2 axis

    # neighbor on the left
    if best_cache2 > 0:
        neighbor_cost = cost(best_thread,   \
                             best_cache1,   \
                             best_cache2-1, \
                             best_cache3)
        if neighbor_cost < best_cost:
            pheromone[best_thread,   \
                      best_cache1,   \
                      best_cache2-1, \
                      best_cache3] += ALPHA*worst_cost/neighbor_cost

    # neighbor on the right
    if best_cache2 < NCACHE2-1:
        neighbor_cost = cost(best_thread,   \
                             best_cache1,   \
                             best_cache2+1, \
                             best_cache3)
        if neighbor_cost < best_cost:
            pheromone[best_thread,   \
                      best_cache1,   \
                      best_cache2+1, \
                      best_cache3] += ALPHA*worst_cost/neighbor_cost

def update_neighbors_cache3():

    # update two neighbors along cache3 axis

    # neighbor on the left
    if best_cache3 > 0:
        neighbor_cost = cost(best_thread, \
                             best_cache1, \
                             best_cache2, \
                             best_cache3-1)
        if neighbor_cost < best_cost:
            pheromone[best_thread, \
                      best_cache1, \
                      best_cache2, \
                      best_cache3-1] += ALPHA*worst_cost/neighbor_cost

    # neighbor on the right
    if best_cache3 < NCACHE3-1:
        neighbor_cost = cost(best_thread, \
                             best_cache1, \
                             best_cache2, \
                             best_cache3+1)
        if neighbor_cost < best_cost:
            pheromone[best_thread, \
                      best_cache1, \
                      best_cache2, \
                      best_cache3+1] += ALPHA*worst_cost/neighbor_cost

def update_probabilities():

    # update probabilities for threads
    sum_pheromone_thread = np.zeros(NTHREAD, dtype=float)
    for i in range(NTHREAD):
        for j in range(NCACHE1):
            for k in range(NCACHE2):
                for l in range(NCACHE3):
                    sum_pheromone_thread[i] += pheromone[i,j,k,l]
    # normalize so probablities add up to 1
    supersum_pheromone_thread = 0
    for i in range(NTHREAD):
        supersum_pheromone_thread += sum_pheromone_thread[i]
    for i in range(NTHREAD):
        probabilities_thread[i] = sum_pheromone_thread[i]/supersum_pheromone_thread


    # update probabilities for cache1
    sum_pheromone_cache1 = np.zeros(NCACHE1, dtype=float)
    for i in range(NCACHE1):
        for j in range(NTHREAD):
            for k in range(NCACHE2):
                for l in range(NCACHE3):
                     sum_pheromone_cache1[i] += pheromone[j,i,k,l]
    # normalize so probablities add up to 1
    supersum_pheromone_cache1 = 0
    for i in range(NCACHE1):
        supersum_pheromone_cache1 += sum_pheromone_cache1[i]
    for i in range(NCACHE1):
        probabilities_cache1[i] = sum_pheromone_cache1[i]/supersum_pheromone_cache1


    # update probabilities for cache2
    sum_pheromone_cache2 = np.zeros(NCACHE2, dtype=float)
    for i in range(NCACHE2):
        for j in range(NTHREAD):
            for k in range(NCACHE1):
                for l in range(NCACHE3):
                     sum_pheromone_cache2[i] += pheromone[j,k,i,l]
    # normalize so probablities add up to 1
    supersum_pheromone_cache2 = 0
    for i in range(NCACHE2):
        supersum_pheromone_cache2 += sum_pheromone_cache2[i]
    for i in range(NCACHE2):
        probabilities_cache2[i] = sum_pheromone_cache2[i]/supersum_pheromone_cache2


    # update probabilities for cache3
    sum_pheromone_cache3 = np.zeros(NCACHE3, dtype=float)
    for i in range(NCACHE3):
        for j in range(NTHREAD):
            for k in range(NCACHE1):
                for l in range(NCACHE2):
                     sum_pheromone_cache3[i] += pheromone[j,k,l,i]
    # normalize so probablities add up to 1
    supersum_pheromone_cache3 = 0
    for i in range(NCACHE3):
        supersum_pheromone_cache3 += sum_pheromone_cache3[i]
    for i in range(NCACHE3):
        probabilities_cache3[i] = sum_pheromone_cache3[i]/supersum_pheromone_cache3


def display_solution():

    global converged

    print("Pheromone of best ant: ")
    print(pheromone[best_thread,best_cache1,best_cache2,best_cache3])
    print("")

    print("Probabilities for thread")
    print(probabilities_thread)
    print("")

    print("Probabilities for cache1")
    print(probabilities_cache1)
    print("")

    print("Probabilities for cache2")
    print(probabilities_cache2)
    print("")

    print("Probabilities for cache3")
    print(probabilities_cache3)
    print("")
    """
    for i in range(NANTS):
        print("Ant number: " + str(i))
        print("   threads[" + str(selected_thread[i]) + "]: " + str(thread[selected_thread[i]]))
        print("   cache1[" + str(selected_cache1[i]) + "]: " + str(cache1[selected_cache1[i]]))
        print("   cache2[" + str(selected_cache2[i]) + "]: " + str(cache2[selected_cache2[i]]))
        print("   cache3[" + str(selected_cache3[i]) + "]: " + str(cache3[selected_cache3[i]]))
        print("   cost: " + str(cost(thread[selected_thread[i]], \
                                     cache1[selected_cache1[i]], \
                                     cache2[selected_cache2[i]], \
                                     cache3[selected_cache3[i]])))
    print("")
    """

    print("Best ant: " + str(best_ant))
    print("Best cost: " + str(best_cost))
    print("Best parameters:")
    print("    thread[" + str(best_thread) + "]: " + str(thread[best_thread])) 
    print("    cache1[" + str(best_cache1) + "]: " + str(cache1[best_cache1]))
    print("    cache2[" + str(best_cache2) + "]: " + str(cache2[best_cache2]))
    print("    cache3[" + str(best_cache3) + "]: " + str(cache3[best_cache3]))
    print("")

    print("Worst ant: " + str(worst_ant))
    print("Worst cost: " + str(worst_cost))
    print("Worst paramters:")
    print("   thread[" + str(worst_thread) + "]: " + str(thread[worst_thread]))
    print("   cache1[" + str(worst_cache1) + "]: " + str(cache1[worst_cache1]))
    print("   cache2[" + str(worst_cache2) + "]: " + str(cache2[worst_cache2]))
    print("   cache3[" + str(worst_cache3) + "]: " + str(cache3[worst_cache3]))
    print("")

    if (best_cost == OPTIMAL_COST):
         print("Optimal path obtained after " + str(iters) + " iterations")
         converged = 1

if __name__ == "__main__":

    global iters

    random.seed(100) # to make results reproducile
    iters = 1

    initialize()

    while (iters < MAXITERS and converged == 0):
        print("ITERATION: " + str(iters))
        print("")
        ants_pick_paths()
        display_solution()
        update_pheromone()
        update_probabilities()
        iters += 1

    
    # Display best solution found
    min_cost = INFINITY
    for i in range(NTHREAD):
        for j in range(NCACHE1):
            for k in range(NCACHE2):
                for l in range(NCACHE3):
                    if solutions[i,j,k,l] < min_cost and solutions[i,j,k,l] > 0:
                        min_cost = solutions[i,j,k,l]
                        min_thread = i
                        min_cache1 = j
                        min_cache2 = k
                        min_cache3 = l
    print("Minimun cost found: " + str(min_cost))
    print("Number of calls to function: " + str(num_calls))
    print("Number of iterations: " + str(iters))
    print("Best parameters found")
    print("   thread[" + str(min_thread) + "]: " + str(thread[min_thread]))
    print("   cache1[" + str(min_cache1) + "]: " + str(cache1[min_cache1]))
    print("   cache2[" + str(min_cache2) + "]: " + str(cache2[min_cache2]))
    print("   cache3[" + str(min_cache3) + "]: " + str(cache3[min_cache3]))
