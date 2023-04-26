import numpy as np
import tensorflow as tf
from tensorflow import keras


def read_datasets():
    # assumes cli has been run to download data
    with open('./data/coursera-data/small_movies_X.csv', 'rb') as f:
        X = np.loadtxt(f, delimiter=',')

    with open('./data/coursera-data/small_movies_W.csv', 'rb') as f:
        W = np.loadtxt(f, delimiter=',')

    with open('./data/coursera-data/small_movies_b.csv', 'rb') as f:
        b = np.loadtxt(f, delimiter=',')

    with open('./data/coursera-data/small_movies_Y.csv', 'rb') as f:
        Y = np.loadtxt(f, delimiter=',')

    with open('./data/coursera-data/small_movies_R.csv', 'rb') as f:
        R = np.loadtxt(f, delimiter=',')

    return X, W, b, Y, R


def manually_compute_cost(X, W, b, Y, R, lambda_) -> float:
    """
    Compute the cost function for collaborative filtering.

    Args:
        X (ndarray): (num_items,num_features) dimensional matrix of features for each item
        W (ndarray): (num_users, num_features) dimensional matrix of features values for each user
        b (ndarray): (1,num_users) dimensional vector of user parameters
        Y (ndarray): (num_movies, num_users) dimensional matrix of user ratings of items
        R (ndarray): (num_movies, num_users) dimensional matrix of booleans indicating if user i rated item j
        lambda_ (float): regularization parameter

    Returns:
        J (float): cost function
    """
    # my implementation
    J = 0
    
    wxb = np.dot(W, X.T) + b.T
    
    err = 0.5*np.sum(R*(wxb.T - Y)**2)
    
    regularization_w = (lambda_ / 2) * np.sum(W**2)
    
    regularization_x = (lambda_ / 2) * np.sum(X**2)
        
    J = err + regularization_w + regularization_x
    return J


def compute_cost(X, W, b, Y, R, lambda_) -> tf.Tensor:
    """
    Compute the cost function for collaborative filtering.

    Args:
        X (ndarray): (num_items,num_features) dimensional matrix of features for each item
        W (ndarray): (num_users, num_features) dimensional matrix of features values for each user
        b (ndarray): (1,num_users) dimensional vector of user parameters
        Y (ndarray): (num_movies, num_users) dimensional matrix of user ratings of items
        R (ndarray): (num_movies, num_users) dimensional matrix of booleans indicating if user i rated item j
        lambda_ (float): regularization parameter

    Returns:
        J (ts.Tensor): cost function
    """

    # coursera (tensorflow) implementation
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    
    return J


if __name__ == "__main__":
    X, W, b, Y, R = read_datasets()
    num_movies, num_features = X.shape
    num_users, _ = W.shape

    num_users_r = 4
    num_movies_r = 5 
    num_features_r = 3

    X_r = X[:num_movies_r, :num_features_r]
    W_r = W[:num_users_r,  :num_features_r]
    b_r = b[:num_users_r].reshape(1, -1)
    Y_r = Y[:num_movies_r, :num_users_r]
    R_r = R[:num_movies_r, :num_users_r]
    J = compute_cost(X_r, W_r, b_r, Y_r, R_r, 0)

    assert np.isclose(J, 13.67, atol=0.01)

    # with regularization
    J = compute_cost(X_r, W_r, b_r, Y_r, R_r, 1.5)
    import pdb; pdb.set_trace()
    assert np.isclose(J, 28.09, atol=0.01)

    
