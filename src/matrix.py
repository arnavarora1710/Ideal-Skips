import numpy as np
from sklearn import preprocessing

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Goal of the function is to fill in the elements of the transition matrix
def computeMatrix(learningArray):
    n = len(learningArray)
    # learningArray = [sigmoid(x) for x in learningArray]
    # learningArray = preprocessing.normalize([learningArray], norm = 'l1')[0]
    # learningArray[0] += 1 - sum(learningArray)
    # print(sum(learningArray))
    matrix = [[0 for i in range(n)] for j in range(n)]
    for i in range(n - 1):
        matrix[i][i + 1] = learningArray[i]; 
    for length in range(2, n):
        for i in range(n):
            j = i + length
            if j < n:
                matrix[i][j] = matrix[i + 1][j] * learningArray[i] + matrix[i][j - 1] * learningArray[j - 1]
    for i in range(n):
        matrix[i] = [sigmoid(x) for x in matrix[i]]
        matrix[i] = preprocessing.normalize([matrix[i]], norm='l1')[0]
        matrix[i][i] += 1 - sum(matrix[i])
        print(sum(matrix[i]))
    return matrix
