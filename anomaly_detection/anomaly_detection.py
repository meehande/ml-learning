import numpy as np
import matplotlib.pyplot as plt



"""
1. read data
2. plot data
3. compute gaussian params
4. compute density function
5. select threshold (epsilon)
    - using F1 score on cross validation set


"""


def compute_gaussian(X: np.array):
    """
    Given set of training examples X, compute the mean and variance of each feature.

        mean_i = 1/m * (sum(x_i))
        var_i = 1/m * sum( x_i - u_i)**2


    Args:
        X (ndarray): (m,n) dimensional matrix of m training examples with n features
    Returns:
        mu (ndarray): (n, ) dimensional array of the mean of each feature from the training set
        var (ndarray): (n, ) dimensional array of the variance of each feature from the training set
    
    
    """

    mu = np.mean(X, axis=0)

    var = np.mean(np.square(X-mu), axis=0)

    return mu, var
    

def multivariate_gaussian_density_function(X: np.array, mu: np.array, var: np.array):
    """
    Given a training set X with the corresponding mean and variance of the features, 
    compute the probability density function
    
    Detailed explanation of formulas: https://cs229.stanford.edu/section/gaussians.pdf
    
    Args:
        X (ndarray): (m,n) array of training examples
        mu (ndarray): (n,) mean of each feature
        var (ndarray): (n, ) variance of each feature
    """

    n = len(mu)

    # if variance is a matrix, treat as covariance matrix
    # else, treat as the variances in each dimension (diagonal covariance matrix)
    if var.ndim == 1:
        var = np.diag(var) 

    X = X - mu

    coefficient = (2*np.pi)**(-n/2) * np.linalg.det(var)**(-0.5)
    power = (-0.5 * np.sum( np.matmul(X, np.linalg.pinv(var)) * X, axis=1))

    p = coefficient * np.exp(power)
    return p



def compute_precision(true_positive, false_positive):
    """
    Of everything classified as positive, what percentage was correct
    (likelihood of a positive result actually being positive)
    
    """
    precision = (true_positive) / (true_positive + false_positive)
    return precision


def compute_recall(true_positive, false_negative):
    """
    Of the actual positives, what percentage were detected 
    (likelihood of missing a positive)
    """
    recall = true_positive / (true_positive + false_negative)
    return recall


def compute_f1(true_positive, false_positive, false_negative):
    precision = compute_precision(true_positive, false_positive)
    recall = compute_recall(true_positive, false_negative)

    return (2* precision * recall) / (precision + recall)


def select_threshold(y_val: np.ndarray, p_val: np.ndarray):
    """
    Based on the probabilities of examples in a validation set (p_val)
    and their corresponding ground truth labels (y_val), 
    find the best threshold (epsilon) to use for selecting outliers
    
    Args:
        y_val (ndarray): Ground truth labels for validation set
        p_val (ndarray): Probabilities for validation set
    Returns: 
        epsilon (float): Threshold chosen
        F1 (float): F1 score of the epsilon chosen
    """

    best_epsilon = 0
    best_f1 = 0
    f1 = 0

    step_size = (max(p_val) - min(p_val)) / 1000

    for current_epsilon in np.arange(min(p_val), max(p_val), step_size):
        predicted_anomalies = (p_val < current_epsilon).astype(int)  # y = 1 for an anomaly - ie less likely sample 
        prediction_matches = predicted_anomalies == y_val
        tp = np.sum((y_val == 1) & prediction_matches)
        fp = np.sum((predicted_anomalies == 1) & ~prediction_matches)
        fn = np.sum((predicted_anomalies == 0) & ~prediction_matches)
        f1 = compute_f1(true_positive=tp, false_positive=fp, false_negative=fn)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = current_epsilon

    return best_epsilon, best_f1


def plot_training_data(X: np.array):
    fig = plt.figure()
    ax = fig.subplots()
    ax.scatter(X[:, 0], X[:, 1], marker="x", c="b")
    fig.show()
    return fig


def plot_data_with_contours(X, mu, var):
    X1, X2 = np.meshgrid(np.arange(0, 35.5, 0.5), np.arange(0, 35.5, 0.5))
    Z = multivariate_gaussian_density_function(np.stack([X1.ravel(), X2.ravel()], axis=1), mu, var)
    Z = Z.reshape(X1.shape)
    
    plt.plot(X[:, 0], X[:, 1], 'bx')

    if np.sum(np.isinf(Z)) == 0:
        plt.contour(X1, X2, Z, levels=10**(np.arange(-20., 1, 3)), linewidths=1)
   
    plt.title("The Gaussian contours of the distribution fit to the dataset")
    

if __name__ == "__main__":
    X_train = np.load("data/X_part1.npy")
    X_validation = np.load("data/X_val_part1.npy")
    y_validation = np.load("data/y_val_part1.npy")

    mu, var = compute_gaussian(X_train)

    assert all(np.isclose(mu, [14.11222578, 14.99771051]))
    assert all(np.isclose(var, [1.83263141, 1.70974533]))

    p = multivariate_gaussian_density_function(X_train, mu, var)

    """
    Use cross validation set to choose epsilon
    """
    p_validation = multivariate_gaussian_density_function(X_validation, mu, var)
    epsilon, F1 = select_threshold(y_validation, p_validation)

    assert np.isclose(epsilon, 8.990853e-05)
    assert np.isclose(F1, 0.875000)

    outliers = p < epsilon

    plot_data_with_contours(X_train, mu, var)
    plt.plot(X_train[outliers, 0], X_train[outliers, 1], 'ro', markersize=10, markerfacecolor='none', markeredgewidth=2)


