import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Tuple


def read_wxb():
    # assumes cli has been run to download data
    with open('./data/coursera-data/small_movies_X.csv', 'rb') as f:
        X = np.loadtxt(f, delimiter=',')

    with open('./data/coursera-data/small_movies_W.csv', 'rb') as f:
        W = np.loadtxt(f, delimiter=',')

    with open('./data/coursera-data/small_movies_b.csv', 'rb') as f:
        b = np.loadtxt(f, delimiter=',')

    return X, W, b, 


def read_ratings():
    with open('./data/coursera-data/small_movies_Y.csv', 'rb') as f:
        Y = np.loadtxt(f, delimiter=',')

    with open('./data/coursera-data/small_movies_R.csv', 'rb') as f:
        R = np.loadtxt(f, delimiter=',')
    return Y, R


def read_movie_list() -> pd.DataFrame:
    df = pd.read_csv('./data/coursera-data/small_movie_list.csv', header=0, index_col=0,  delimiter=',', quotechar='"')
    mlist = df["title"].to_list()
    return (mlist, df)


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


def train_model(X, W, b, Y_norm, R, lambda_, alpha,) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    
    """
    optimized = tf.keras.optimizers.Adam(learning_rate=alpha)
    iterations = 200
    for i in range(iterations):
        with tf.GradientTape() as tape:
            J = compute_cost(X, W, b, Y_norm, R, lambda_)
        grads = tape.gradient(J, [X, W, b])
        optimized.apply_gradients(zip(grads, [X, W, b]))
        if i % 10 == 0:
            print(f'iteration {i}, cost: {J}')    
    return X, W, b


def predict(X: tf.Tensor, W: tf.Tensor, b: tf.Tensor, Y_mean: np.array):
    p = np.matmul(X.numpy(), W.numpy().T) + b.numpy().T

    p = p + Y_mean
    return p


def _validate_cost_fn():
    X, W, b = read_wxb()
    Y, R = read_ratings()
    num_movies, num_features = X.shape
    num_users, _ = W.shape

    # take sample 
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

    J = compute_cost(X_r, W_r, b_r, Y_r, R_r, 1.5)
    assert np.isclose(J, 28.09, atol=0.01)


def mean_normalize_ratings(Y, R):
    # mean normalization for collaboartive filtering just subtracts mean 
    # to each rating
    # Predict then needs to reverse this -> thus returning the mean
    # for each item for users who have not rated anything 
    # (helps with cold start problem, on top of increasing efficiency of gradient descent)  
    # NB: use R to ensure correct divisor -> only include items a user has rated in computing mean
    Ymean = (np.sum(Y*R,axis=1)/(np.sum(R, axis=1)+1e-12)).reshape(-1,1)
    Ynorm = Y - np.multiply(Ymean, R)
    return Ynorm, Ymean



if __name__ == "__main__":
    Y, R = read_ratings()    
    num_movies = Y.shape[0]

    my_y = np.zeros(num_movies)
    # let copilot rate movies..!
    my_y[0] = 4
    my_y[97] = 2
    my_y[6] = 3
    my_y[11]= 5
    my_y[53] = 4
    my_y[63]= 5
    my_y[794]= 3
    my_y[183]= 4
    my_y[226] = 5

    my_r = (my_y > 0).astype(int)

    Y = np.c_[Y, my_y]
    R = np.c_[R, my_r]

    Ymean, Ynorm = mean_normalize_ratings(Y, R)

    num_items, num_users = Y.shape
    num_features = 100

    tf.random.set_seed(1234) # for consistent results
    W = tf.Variable(tf.random.normal((num_users, num_features),dtype=tf.float64),  name='W')
    X = tf.Variable(tf.random.normal((num_movies, num_features),dtype=tf.float64),  name='X')
    b = tf.Variable(tf.random.normal((1, num_users),   dtype=tf.float64),  name='b')

    train_model(X, W, b, Ynorm, R, 1, 1e-1)

    predictions = predict(X, W, b, Ymean)

    
        





    


