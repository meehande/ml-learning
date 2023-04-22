import numpy as np

def find_closest_centroid(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Finds the closest centroid for each training example (closeness defined by L2-norm)

    Args:
        X (ndarray): (m, n) m input training samples with n feature dimensions
        centroids (ndarray): (K, n) n-dimensional coordinates of K cluster centres
    Returns: 
        idx (ndarray): (m, ) id in centroids of the closest centroid for each training sample in X

    """
    m, _ = X.shape
    idx = np.zeros(m)

    for i in range(m):
        distance = np.linalg.norm(centroids - X[i], axis=1)
        idx[i] = np.argmin(distance)
    return idx


def compute_centroids(X: np.array, idx: np.array, K: int):
    """
    Returns the coordinates of the K cluster centroids determined by the coordinates
    of training examples (X) and their cluster assignments (idx)

    Args: 
        X (ndarray): (m, n) m input training samples with n feature dimensions
        idx (ndarray): (m, ) id in centroids of the closest centroid for each training sample in X
        K (int): the number of clusters 
    Returns: 
        centroids (ndarray): (K, n) n-dimensional coordinates of K cluster centres
    
    """
    _, n = X.shape
    centroids = np.zeros((K, n))

    for i in range(K):
        centroids[i] = np.mean(X[idx==i], axis=0)
    return centroids


def initialise_centroids(X, K):
    """
    Define initial K cluster centroids by choosing randomly from the training examples (X)
    
    """
    idxs = np.random.permutation(X.shape[0])[:K] # permutation because we don't want duplicates
    return X[idxs]


def run_kmeans(X, initial_centroids, max_iters=10, plot_progress=False):
    centroids = initial_centroids
    K = initial_centroids.shape[0]
    for i in range(max_iters):
        idx = find_closest_centroid(X, centroids)
        centroids = compute_centroids(X, idx, K)
        # todo: plot data!

    return centroids, idx



if __name__ == "__main__":
    X = np.array([[1.84207953, 4.6075716 ],
            [5.65858312, 4.79996405],
            [6.35257892, 3.2908545 ],
            [2.90401653, 4.61220411],
            [3.23197916, 4.93989405],])

    centroids = np.array([[3,3], [6,2], [8,5]])

    idx = find_closest_centroid(X, centroids)
    assert all(idx == np.array([0., 2., 1., 0., 0.]))

    new_centroids = compute_centroids(X, idx, K=3)




