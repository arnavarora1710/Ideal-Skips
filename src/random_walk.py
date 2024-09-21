import numpy as np

def simple_random_walk(n, start, transition_matrix, steps):
    result = [start]
    choicer = []
    for i in range(n):
        choicer.append(i)
    for i in range(steps):
        choice = np.random.choice(choicer, p = transition_matrix[start])
        result.append(choice)
        start = choice
    return result

WINDOW_SIZE = 3

# define a distance metric to penalize jumps close to each other
# dont want to skip i->j and i+1->j+1 etc
def update(i, j, k, l, p_kl):
    # now need to convert to prob
    # TODO: test with other ways to convert except for sigmoid
    euclidean_dist = np.sqrt((k - i) * (k - i) + (l - j) * (l - j))
    return max(0, p_kl - (1 - np.sigmoid(euclidean_dist)))

def valid(i, j, n, m):
    return i >= 0 and j >= 0 and i < n and j < m
    
def random_walk_with_updates(n, start, transition_matrix, steps):
    result = [start]
    choicer = [x for x in range(n)]
    N = len(transition_matrix)
    M = len(transition_matrix[0])
    for _ in range(steps):
        choice = np.random.choice(choicer, p = transition_matrix[start])
        result.append(choice)
        edge = [start, choice]
        for c1 in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
            for c2 in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
                i = edge[0]
                k = i + c1
                j = edge[1]
                l = j + c2
                if valid(k, l, N, M):
                    transition_matrix[k][l] = update(i, j, k, l, transition_matrix[k][l])
        start = choice
    return result