import numpy as np

def random_walk(n, start, transition_matrix, steps):
    result = [start]
    choicer = []
    for i in range(n):
        choicer.append(i)
    for i in range(steps):
        choice = np.random.choice(choicer, p = transition_matrix[start])
        result.append(choice)
        start = choice
    return result