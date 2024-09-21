
import numpy as np

# A function to find the sum between i and j to create Lij
def sumBtw(prefixLearning, i, j):
    return prefixLearning[j] - prefixLearning[i]


# A function to find and return the norm of the weight i and j [(w_i - w_j)]
def findWt(wt_i, wt_j):
    norm_vec1 = np.linalg.norm(wt_i)
    norm_vec2 = np.linalg.norm(wt_j)
    
    # Return the absolute difference between the norms
    return abs(norm_vec1 - norm_vec2)

# Goal of the function is to fill in the elements of the transition matrix
def computeMatrix(learningArray, weights):

    # define the transition matrix
    n = len(learningArray)
    matrix = [[0] * n]

    # Compute the prefix sum of the learningMatrix
    prefixLearning = [0]

    for i in range(1, n):
        prefixLearning[i] = prefixLearning[i - 1] + learningArray[i]
    

    # Fill the indices in the graph 
    for i in range(0, n):
        for j in range(i + 1, n):
            matrix[i][j] = ((findWt(weights[i], weights[j])) / (j - i)) / sumBtw(prefixLearning, i, j)
    
    return matrix
