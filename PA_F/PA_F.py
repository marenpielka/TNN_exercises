# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 12:39:53 2019

@author: Maren
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


'''
    For the data from task 64, a linear kernel shows to be most appropriate.
    The support vectors are [2, 2] and [3, 1].
    Changing C does not affect the result, because there are no misclassified samples,
    and thus no use for the regularization.
    
    For the training data from files PA-F_t2 to t6, a linear kernel is the optimal choice
    as well, since the data is perfectly linearly separable.
    
    In case of PA-F_t7, an RBF kernel yields the optimal result, since the positive points
    form a circle surrounded by the negative points. This problem is not linearly separable.
    A larger C will result in a greater radius of the area surrounding the positive
    examples (for the RBF kernel).

    Visualization code taken from an online sklearn example
    (https://scikit-learn.org/stable/auto_examples/svm/plot_iris.html)
'''


def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def visualize_svm_results(models, titles, X, y):
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    
    for clf, title, ax in zip(models, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy,
                      cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
    
    plt.show()
    
    
def read_data_from_file(filename):
    with open(filename, 'r') as fp:
        lines = list(fp)
        X = []
        y = []
    
        # Read input vectors
        i = 0
        for line in lines:
            if i != 0:
                l = list(filter(None, line.split(' ')))
                input_vector = [float(x.split(":")[1]) for x in l[1:]]
                X.append(input_vector)
                y.append(float(l[0]))
            i += 1
                
    return X, y


def main():
    # input data from task 64
    X = np.array([[0.0, 3.0], [1.0, 3.0], [1.0, 2.0], [2.0, 2.0], [2.0, 4.0],\
                  [5.0, 8.0], [0.0, -3.0], [1.0, -3.0], [1.0, -2.0], [2.0, -4.0],\
                  [3.0, 1.0],[3.0, 0.0], [3.0,-2.0], [4.0, -1.0], [5.0, 1.0]])
    y = np.concatenate((np.zeros(6), np.ones(9)))
    
    '''    
    # Read input data from file
    X, y = read_data_from_file('PA-F_t7.dat')
    X = np.array(X)
    y = np.array(y)
    '''
    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 2.0  # SVM regularization parameter
    models = (svm.SVC(kernel='linear', C=C),
              svm.LinearSVC(C=C),
              svm.SVC(kernel='rbf', gamma=0.7, C=C),
              svm.SVC(kernel='poly', degree=3, C=C))
    models = (clf.fit(X, y) for clf in models)
    
    # title for the plots
    titles = ('SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel')
    
    '''
    for model in models:
        print(model.support_vectors_)
        break
    '''
    visualize_svm_results(models, titles, X, y)
    

if __name__ == "__main__":
    main()
