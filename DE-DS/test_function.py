import numpy as np
import math
def Sphere(ind):
    sum = 0
    for i in ind:
        sum += i**2
    return sum

def Restrigin(ind):
    sum = 10 * len(ind)
    for i in ind:
        sum += i**2 - 10 * np.cos(2*np.pi*i)
    return sum 

def Rosenbrock(ind):
    sum = 0
    for i in range(len(ind) - 1):
        sum += 100 * (ind[i + 1] - ind[i]**2)**2 + (ind[i] - 1)**2
    return sum 

def Griewank(ind):
    sum_1 = 0
    for i in ind:
        sum_1 += (i**2)/4000
    sum_2 = 1
    for i in range(len(ind)):
        sum_2 *= np.cos(ind[i]/math.sqrt(i + 1)) + 1
    return sum_1 - sum_2

def Ackley(ind):
    a, b, c = 20, 0.2, 2*np.pi
    sum_1 = 0
    for i in ind:
        sum_1 += i**2
    sum_1 = -1 * a * np.exp(-b * math.sqrt(sum_1 * (1/len(ind))))
    
    sum_2 = 0
    for i in ind:
        sum_2 += np.cos(c*i)
    sum_2 = np.exp((1/len(ind)) * sum_2)
    return sum_1 - sum_2 + a + np.exp(1)
    
    