# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:39:18 2014

@author: Alejandro
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from random import shuffle
from numpy.random import uniform

class GaussianExperiment(object):
    def __init__(self, n_samples, n_outliers, n_clusters, n_features, n_experiment):
        self.n_samples = n_samples
        self.n_outliers = n_outliers        
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.n_experiment = n_experiment
        
    def show_graph_3d(self, X,y,X_contaminated):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.m = 'o'
        self.ax.scatter(X[:,0], X[:,1], X[:,2], c=y, marker=self.m)
        self.ax.scatter(X_contaminated[:,0], X_contaminated[:,1], X_contaminated[:,2], c=['g' for i in range(X_contaminated[:,0].shape[0])], marker='^')
        self.ax.set_xlabel('First Dimension')
        self.ax.set_ylabel('Second Dimension')
        self.ax.set_zlabel('Third Dimension')
        plt.show()
    
    def generate_data(self):
        if self.n_experiment == 1:
            self.X, self.y = make_blobs(n_samples=300, centers=self.n_clusters, n_features=3,random_state=3)
        if self.n_experiment == 2:
            self.X, self.y = make_blobs(n_samples=300, centers=self.n_clusters, n_features=3,random_state=5)
        return self.X,self.y
    
    def generate_contamination(self):
        if self.n_experiment == 1:
            self.contamination_total = np.zeros((self.n_outliers,self.n_features))
            for i in range (self.n_outliers):
                list_partition = [x for x in range(self.n_features)]
                shuffle(list_partition)
                dimension = np.zeros((3))
                dimension[0] = uniform(low=-10,high=-4,size=1)
                dimension[1] = uniform(low=-4,high=4,size=1)
                dimension[2] = uniform(low=4,high=10,size=1)
                self.contamination_total[i,:] = np.array(([dimension[list_partition[0]],dimension[list_partition[1]],dimension[list_partition[2]]]))        
            return self.contamination_total
            
        if self.n_experiment == 2:
            self.contamination_total = np.zeros((self.n_outliers,self.n_features))
            for i in range (self.n_outliers):
                list_partition = [x for x in range(self.n_features)]
                shuffle(list_partition)
                dimension = np.zeros((3))
                dimension[0] = uniform(low=-10,high=0,size=1)
                dimension[1] = uniform(low=-10,high=5,size=1)
                dimension[2] = uniform(low=-3,high=10,size=1)
                self.contamination_total[i,:] = np.array(([dimension[list_partition[0]],dimension[list_partition[1]],dimension[list_partition[2]]]))        
            return self.contamination_total            
#np.savez('Data/data_experiment_1', mean =  ,data = data[[1,3,5]][0], contamination = contamination)