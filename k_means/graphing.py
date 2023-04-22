import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D




def plot_2D_kmeans(X, cluster_assignments, centroids, previous_centroids, iteration, fig=None, ax=None):
    if fig is None:
        fig = plt.figure()
        ax = fig.subplots()

        colormap = ListedColormap(["red", "green", "blue"]) # todo: this should be based on # centroids

        color_assignments = colormap(cluster_assignments)

        ax.scatter(
            X[:, 0], 
            X[:,1], 
            facecolors='none', 
            edgecolors=color_assignments, 
            linewidth=0.1,
            alpha=0.7
            )

    ax.scatter(centroids[:,0], centroids[:,1],  marker='x', c='k', linewidths=3)

    # plot line from previous centroid to this one to see where it moved
    ax.plot([centroids[0,0], previous_centroids[0,0]], [centroids[0,1], previous_centroids[0,1]], '-k', 1)
    ax.plot([centroids[1,0], previous_centroids[1,0]], [centroids[1,1], previous_centroids[1,1]], '-k', 1)
    ax.plot([centroids[2,0], previous_centroids[2,0]], [centroids[2,1], previous_centroids[2,1]], '-k', 1)

    #ax.title("Iteration number %d", iteration)
    return fig, ax


f = None
a = None
f, a = plot_2D_kmeans(X, idx, centroids, initial_centroids, 1, f, a)


"""
TODO: 
- make the chart show how the cluster centroid moves
(manually by computing all the data and adding all the lines)
-> then figure out how matplotlib works enough to do it by updating one figure / axis on each iteration
"""