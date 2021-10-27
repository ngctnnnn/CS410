import numpy as np
import matplotlib.pyplot as plt
from test_function import Griewank, Rosenbrock, Sphere, Ackley, Rastrigin
from constant import seed_number, bound_lower, bound_upper, max_evals, F_scale, cross_prob
from constant import sigma_init, c_inc, c_dec, popsize, max_evals, dimension
from algorithm import DE, ES
from celluloid import Camera
    
def DifferentialEvolution(test_function, dimension, bound_lower, bound_upper, F_scale, cross_prob, popsize, max_evals):
    np.random.rand(seed_number)
    
    results, all_pops, generation_count = DE(test_function, dimension, [(bound_lower, bound_upper)]*dimension, F_scale, cross_prob, popsize, max_evals)
    bound_lower = -6
    bound_upper = 6
    x = np.linspace(bound_lower, bound_upper, 100)
    y = np.linspace(bound_lower, bound_upper, 100)
    X, Y = np.meshgrid(x, y)
    Z = test_function([X, Y])
    
    
    fig = plt.figure(figsize=(12, 12))
    camera = Camera(fig)
    plt.contourf(X, Y, Z, popsize, cmap='viridis')
    plt.axis('square')
    plt.scatter(0, 0, marker='*')
    
    for generation in range(generation_count):
        plt.contourf(X, Y, Z, popsize, cmap='viridis')
        plt.axis('square')
        plt.scatter(0, 0, marker='*')
        plt.scatter(all_pops[generation][:, 0], all_pops[generation][:, 1], c='#ff0000', marker='o')
        plt.plot()
        plt.pause(0.01)
        camera.snap()
    plt.show()
    anim = camera.animate()
    #save the animation as a gif file
    anim.save('DE/Griewank' + "/" + 'Griewank' + "-DE-" + str(popsize) + ".gif",writer="pillow")
    
def EvolutionStrategies(test_function, bounds, sigma_init, c_inc, c_dec, popsize, max_evals, dimension):
    np.random.rand(seed_number)
    all_pops = []
    generation_count = 0
    results = []
    results, all_pops, generation_count = ES(test_function, bounds, sigma_init, c_inc, c_dec, popsize, max_evals, dimension)
    
    bound_lower = -33
    bound_upper = 33
    x = np.linspace(bound_lower, bound_upper, 100)
    y = np.linspace(bound_lower, bound_upper, 100)
    X, Y = np.meshgrid(x, y)
    Z = test_function([X, Y])
    
    fig = plt.figure(figsize=(12, 12))
    camera = Camera(fig)
    plt.contourf(X, Y, Z, popsize, cmap='viridis')
    plt.axis('square')
    plt.scatter(0, 0, marker='*')
    
    for generation in range(generation_count):
        plt.contourf(X, Y, Z, popsize, cmap='viridis')
        plt.axis('square')
        plt.scatter(0, 0, marker='*')
        plt.scatter(all_pops[generation][:, 0], all_pops[generation][:, 1], c='#ff0000', marker='o')
        plt.plot()
        plt.pause(0.01)
        camera.snap()
    plt.show()
    anim = camera.animate()
    
    #save the animation as a gif file
    anim.save('ES/Rastrigin' + "/" + 'Rastrigin' + "-ES-" + str(popsize) + ".gif",writer="pillow")
 
if __name__=='__main__':
    all_fitness = []
    num_evaluation = []
    
    test_function = Griewank
    seed_number = 19520925
    
    popsize_array = [64, 128, 256, 512, 1024]
    for popsize in popsize_array:
        print(f"Popsize = {popsize}")
        if dimension == 2:
            max_evals = 1e5
            bounds = [(bound_lower, bound_upper)]*dimension
            mean_result = np.mean(all_fitness)
            stddev_result = np.std(all_fitness)
            mean_evaluation = np.mean(num_evaluation)
            
            DifferentialEvolution(test_function, dimension, bound_lower, bound_upper, F_scale, cross_prob, popsize, max_evals)
            # EvolutionStrategies(test_function, bounds, sigma_init, c_inc, c_dec, popsize, max_evals, dimension)