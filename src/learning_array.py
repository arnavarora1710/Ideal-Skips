import numpy as np
from sklearn.preprocessing import MinMaxScaler
from npeet import entropy_estimators as ee

# returns learning array with prefix sum on that array
def gen_learning_array(weight_matrices):
    # L[i] = learning between i -> i + 1
    N = len(weight_matrices)
    L = [0.0] * (N - 1)
    scaler = MinMaxScaler()
    for i in range(N - 1):
        out1 = weight_matrices[i]
        out2 = weight_matrices[i + 1]
        out1_np = out1.detach().numpy()
        out2_np = out2.detach().numpy()
        out1_np = scaler.fit_transform(out1_np)
        out2_np = scaler.fit_transform(out2_np)
        # compute mutual information gain between layer i and layer i + 1
        mi = ee.mi(out1_np, out2_np)
        L[i] = mi
    L = [x / float(sum(L)) for x in L]
    return L