import numpy as np
import matplotlib.pyplot as pl
from test_function import Griewank, Rosenbrock, Sphere, Ackley, Restrigin


def initialize_population( num_individuals, num_variables ):
    """
    Khởi tạo quần thể gồm num_individuals cá thể. Mỗi cá thể có num_parameters biến.
    
    Arguments:
    num_individuals -- Số lượng cá thể
    num_variables -- Số lượng biến
    
    Returns:
    pop -- Ma trận (num_individuals, num_variables) chứa quần thể mới được khởi tạo ngẫu nhiên.
    """
    
    pop = np.random.randint(2, size=(num_individuals, num_variables))
    
    return pop

def fitness_evaluation(pop, test_function_number):
    """
    Tính độ thích nghi của quần thể
    Args: 
    pop -- quần thể đang xét
    test_function_number -- hàm cần tính 
    
    1: Sphere
    2: Restrigin
    3: Rosenbrock
    4: Griewank
    5: Ackley
    
    Returns:
    fitness -- Độ thích nghi của quần thể
    """
    if test_function_number == 1:
        values = np.array(Sphere(ind) for ind in pop)
    if test_function_number == 2:
        values = np.array(Restrigin(ind) for ind in pop)
    if test_function_number == 3:
        values = np.array(Rosenbrock(ind) for ind in pop)
    if test_function_number == 4:
        values = np.array(Griewank(ind) for ind in pop)
    if test_function_number == 5:
        values = np.array(Ackley(ind) for ind in pop)
    return values 

def better_fitness( fitness_1, fitness_2, maximization=True ):
    """
    Hàm so sánh độ thích nghi của 2 cá thể.
    
    Arguments:
    fitness_1 -- Độ thích nghi của cá thể 1.
    fitness_2 -- Độ thích nghi của cá thể 2.
    maximization -- Biến boolean cho biết bài toán đang giải thuộc dạng tối đa hoá (mặc định) hay không
    
    Returns:
    True nếu fitness_1 tốt hơn fitness_2. Ngược lại, trả về False.
    """
    
    if maximization:
        if fitness_1 > fitness_2:
            return True
    else:
        if fitness_1 < fitness_2:
            return True
        
    return False

