import numpy as np

def DE(test_function, dimension, bounds, F_scale, cross_prob, popsize, max_evals):
    """
    Differential Evolution algorithm

    Args:
    test_function -- function to conduct
    bound_lower -- lower bound of the test function
    bound_upper -- upper bound of the test function
    F_scale -- scale factor on mutation
    cross_prob -- the probability of 2 individuals to do crossover
    popsize -- the population size
    max_evals -- the maximum fitness evaluation for the algorithm
    seed_number -- value of seed we want to run

    Returns:
    results -- best results after finishing the algorithm
    all_pops -- all the population 
    """
    eps = 0.00001

    bound_lower, bound_upper = np.asarray(bounds).T

    diff = np.fabs(bound_lower - bound_upper)

    pop = bound_lower + diff * np.random.rand(popsize, dimension)

    fitness = np.asarray([test_function(ind) for ind in pop])
    num_eval = 1
    
    best_idx = np.argmin(fitness)
    best = pop[best_idx]

    results = []
    all_pops = []
    results.append((np.copy(best), fitness[best_idx], num_eval))
    all_pops.append(np.copy(pop))
    generation_count = 0
    
    while True:
        # max_evals = 10000 if popsize >= 512 else 5000
        if num_eval > max_evals:
            break
        for i in range(popsize):
            # Mutation step
            idxes = [idx for idx in range(popsize) if idx != i]
            a, b, c = pop[np.random.choice(idxes, 3, replace=False)]
            mutant = np.clip(F_scale*(b - c) + a, bound_lower, bound_upper)

            # Create cross point
            cross_points = np.random.rand(dimension) < cross_prob
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimension)] = True
            
            # Offspring
            trial = np.where(cross_points, mutant, pop[i])

            # Evaluate fitness
            f = test_function(trial)
            num_eval += 1

            if f < fitness[i]:
                pop[i] = trial
                fitness[i] = f 
                if f < fitness[best_idx]:
                    best = trial
                    best_idx = i

        results.append((np.copy(best), fitness[best_idx], num_eval))
        all_pops.append(np.copy(pop))

        if test_function(best) < eps:
            num_eval += 1
            break

        generation_count += 1

    return results, all_pops, generation_count


def ES(test_function, bounds, sigma_init, c_inc, c_dec, popsize, max_evals, dimension):
    eps = 0.00001

    bound_lower, bound_upper = np.asarray(bounds).T

    diff = np.fabs(bound_lower - bound_upper)

    mu = bound_lower + diff * np.random.rand(dimension)
    mu_fitness = test_function(mu)
    num_eval = 0

    results = []
    all_pops = []
    results.append((np.copy(mu), mu_fitness, num_eval))
    generation_count = 0
    sigma = sigma_init
    
    while True:
        max_evals = 10000 if popsize >= 512 else 5000
        if num_eval > max_evals:
            break
        epsilon = np.random.randn(popsize, dimension)
        offspring = mu + sigma * epsilon
        offspring = np.clip(offspring, bound_lower, bound_upper)
        offspring_fitness = np.asarray([test_function(offspring[i]) for i in range(popsize)])
        num_eval += popsize
        
        best_idx = offspring_fitness.argmin()
        best_fitness = offspring_fitness[best_idx]
        best_offspring = offspring[best_idx]

        if best_fitness <= mu_fitness:
            mu = best_offspring.copy()
            mu_fitness = best_fitness
            sigma *= c_inc 
        else:
            sigma *= c_dec
        
        results.append((np.copy(mu), mu_fitness, num_eval))
        all_pops.append(np.copy(offspring))
        if abs(mu_fitness) < eps:
            break
        generation_count += 1

    return results, all_pops, generation_count

    