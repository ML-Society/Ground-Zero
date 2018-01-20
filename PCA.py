import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import csv

''''# no. datapoints in dataset
num_datapoints = 100

# no. dimensions in original data
dims1 = 3

# means
means = np.zeros((dims1))
print(means)

# covariance matrix
cov = np.random.rand(dims1,dims1)
print('cov', cov)

# n dimensional data
x = np.random.multivariate_normal(means, cov, num_datapoints).T
#data = pd.DataFrame(x)
'''

data = datasets.load_iris()

X = data.data
Y = data.target
print(Y)

print(X.shape)
print(Y.shape)

num_datapoints = X.shape[0]
print('number of datapoints: ', num_datapoints)
print('original', X)

# standardise the data
def normalise(x):
    """returns normalised design matrix"""
    # subtract mean
    x_std = x - np.mean(x,axis=0)
    # divide by standard deviation
    x_std = np.divide(x_std, np.std(x_std, axis=0))
    return x_std


def decompose(x):
    """Returns the eigendecomposition of the design matrix?????"""
    # compute covariance matrix of standardised design matrix
    cov = np.matmul(X_std.T, X_std)
    print('Covariance matrix')
    print(cov)

    # compute unit eigenvectors and corresponding eigenvalues
    # the eig function returns the eigens in decreasing order as we want
    # the eigenvalues represent by how much the
    eig_vals, eig_vecs = np.linalg.eig(cov)
    print('eigenvectors - each column represents a unit eigenvector')
    print(eig_vecs)
    print('corresponding eigenvalues')
    print(eig_vals)
    return eig_vals, eig_vecs, cov

def whicheigs(eig_vals):
    """Produces a plot to show the importance of each eigenvector"""
    total = sum(eig_vals)
    var_percent = [(i/total)*100 for i in sorted(eig_vals, reverse=True)]
    cum_var_percent = np.cumsum(var_percent)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('Variance along different principal components')
    ax.grid()
    plt.xlabel('Principal Component')
    plt.ylabel('Percentage total variance accounted for')
    ax.plot(cum_var_percent,'-ro')
    ax.bar(range(len(eig_vals)), var_percent)
    plt.xticks(np.arange(len(eig_vals)), ('PC{}'.format(i) for i in range(len(eig_vals))))
    plt.show

'''
# alternatively we can decompose the matrix via SVD
u, s, v = np.linalg.svd(X_std)

'''

# now we want to project the original data onto these eigenvectors
# by choosing the greatest r eigenvalues we can reduce the dimensions
# we can reduce the number of dimensions that make up the data
# this can reduce the data we have to store and allow us to see
# the principal components of the data - the variables that really matter

def reduce(x, eig_vecs, dims):
    """Returns the dimensionally reduced design matrix and the reducing transform"""
    # the transform consists of the dims most significant eigenvectors
    W = eig_vecs[:,:dims]
    print('Dimension reducing matrix: ')
    print(W)
    return np.matmul(x, W), W

def plotreduced(x):
    dims = x.shape[1]
    try:
        assert dims<= 3, "Project into a dimension between 1 and 3 to visualise"
        fig = plt.figure()
        if dims == 3:
            ax = fig.add_subplot(111, projection='{}d'.format(x.shape[1]))
            ax.scatter(x[:,0], x[:,1], x[:,2], c=colour_list)
            plt.xlabel('PC1 value')
            plt.ylabel('PC2 value')
            ax.set_zlabel('PC3 value')
        elif dims == 2:
            ax = fig.add_subplot(111)
            ax.scatter(x[:,0], x[:,1], c=colour_list)
            plt.xlabel('PC1 value')
            plt.ylabel('PC2 value')
        else:
            ax = fig.add_subplot(111)
            ax.scatter(x, np.zeros(len(x)), c=colour_list)
            plt.xlabel('PC1 value')
        plt.title('Data reduced to {} dimensions'.format(x.shape[1]))
    except AssertionError:
        pass
    plt.grid()
    plt.show()

colour_dict = {0:'r', 1:'g', 2:'b'}
colour_list = [colour_dict[i] for i in list(Y)]

# call all of our functions to actually do something
X_std = normalise(X)
eig_vals, eig_vecs, covariance = decompose(X_std)
whicheigs(eig_vals)
X_reduced, transform = reduce(X_std, eig_vecs, 3)
print(X_reduced.shape)
print(transform.shape)
plotreduced(X_reduced)

def normalise_and_transform(x, transform):
    normalise(x)
    return np.matmul(transform, x)