#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:44:31 2018

@author: remussn
"""
import numpy as np
from grakel import GraphKernel
from sklearn.svm import SVR
# CP from graph_kernels.py
kernel_names_mini = ["propagation", "edge_histogram"]
kernel_names_default = ["subtree_wl", "random_walk",
    "shortest_path",
    # "graphlet_sampling",
    #"subgraph_matching",
    # "multiscale_laplacian",
    # "lovasz_theta", "svm_theta",
    # "neighborhood_hash", "neighborhood_subgraph_pairwise_distance",
    #"NSPDK",
    "odd_sth", "propagation",
    "pyramid_match",
    "propagation", "vertex_histogram", "edge_histogram",
    # "graph_hopper",
    # "weisfeiler_lehman"
    ]


class KernelSVR(object):
    def __init__(self, kernel_names=None):
        #reg = SVR(kernel='precomputed')
        #reg.fit(K_train, my_train)
        # self.kernel_names
        if kernel_names == None:
            self.kernel_names = kernel_names_default
        else:
            self.kernel_names = kernel_names
            
        self.kernel_functions = self._get_kernel_functions()
        
            
    def _get_kernel_functions(self):
        kernel_list = []
        for kernelIdx in range(len(self.kernel_names)):            
            # kernel_list.append(GraphKernel(kernel = [{"name": self.kernel_names[kernelIdx]}], normalize=True))
            kernel_list.append(GraphKernel(kernel = [{"name": self.kernel_names[kernelIdx]}], normalize=False))
        return kernel_list
    
    def add_kernel(self, kernel):
        self.kernel_functions.append(kernel)
    
#    def _get_SVRs(self):
#        svr_list = []
#        for svrIdx in range(len(self.kernel_functions)):
#            svr_list.append(SVR(kernel='precomputed'))
#        return svr_list
    
    def _listDif(self, l1, l2):
        # return sum([(np.float64(e1)-np.float64(e2))^2 for (e1,e2) in zip(l1,l2)])
        return sum([(np.float64(e1)-np.float64(e2))**2 for (e1,e2) in zip(l1,l2)])
    
    def _listNorm(self,l):
        return [li/sum(l) for li in l]
    
    def _get_SVR_weight(self, loss_list, mode='fraction'):
        if mode == 'fraction':
            return np.array(self._listNorm([1/loss for loss in loss_list]))
        elif mode == 'softmax':
            loss_arr = np.array(loss_list)
            return np.exp(-loss_arr)/np.sum(np.exp(-loss_arr))

    def fit_kernel(self, graph_list):
        print('Computing kernels...')
        self.kernels = []
        for i in range(len(self.kernel_functions)):
            print('kernel: '+ self.kernel_names[i])
            self.kernels.append(np.nan_to_num(self.kernel_functions[i].fit_transform(graph_list)))
        # self.kernels = [kernel_function.fit_transform(graph_list) for kernel_function in self.kernel_functions]
    
    def fit_transform_kernel(self, graph_list):
        self.fit_kernel(graph_list)
        return self.kernels
    
    def fit_SVRs(self, y_list):
        if len(self.kernels) == 0:
            print('Please fit kernels first!')
            return
        print('Fitting SVRs...')
        self.SVRs = [SVR(kernel='precomputed').fit(kernel, y_list) for kernel in self.kernels]
        self.SVRs_pred_train = [self.SVRs[idx].predict(self.kernels[idx]) for idx in range(len(self.kernels))]        
        self.SVRs_loss_train = [self._listDif(SVR_pred, y_list) for SVR_pred in self.SVRs_pred_train]
        self.SVRs_weight = self._get_SVR_weight(self.SVRs_loss_train, 'fraction')

    def fit(self, graph_list, y_list):        
        # fit all the kernels and SVRs
        self.fit_kernel(graph_list)
        self.fit_SVRs(y_list)
#        print('Computing kernels...')
#        self.kernels = [kernel_function.fit_transform(graph_list) for kernel_function in self.kernel_functions]
#        print('Fitting SVRs...')
#        self.SVRs = [SVR(kernel='precomputed').fit(kernel, y_list) for kernel in self.kernels]
#        SVRs_pred_train = [self.SVRs[idx].predict(self.kernels[idx]) for idx in range(len(self.kernels))]        
#        SVRs_loss_train = [self._listDif(SVR_pred-y_list) for SVR_pred in SVRs_pred_train]
#        self.SVRs_weight = np.array(self._listNorm([1/loss for loss in SVRs_loss_train]))
#        print('Fitting finished!')
        # self.SVRs_weights = 
        # for kernelIdx in range(len(self.kernel_functions)):
        #     kernel = self.kernel_functions[kernelIdx].fit_transform(graph_list)
        #     self.kernels.append(kernel)            
        #     self.SVRs.append(SVR(kernel='precomputed').fit(kernel, y_list))
    
    def transform(self, graphs_list):
        pass
    
    def fit_transform(self, graphs_list, y_list):
        self.fit(graphs_list, y_list)
        return self.predict(graphs_list)
    
    def predict(self, graph_list, mode='weighted_average'):
        # Compute kernels
        print('Computing kernels...')
        kernels_pred = [np.nan_to_num(kernel_function.transform(graph_list)) for kernel_function in self.kernel_functions]
        # prediction of all SVRs
        print('SVRs Predicting...')
        SVRs_preds = [self.SVRs[idx].predict(kernels_pred[idx]) for idx in range(len(self.kernels))] # List of ndarraies
        if mode == 'all':
            return SVRs_preds
        elif mode == 'weighted_average':
            SVRs_preds_weighted = [weight*pred for (weight, pred) in zip(self.SVRs_weight, SVRs_preds)]
            # SVRs_pred_weighted = np.average(SVRs_preds_weighted, axis=0)
            SVRs_pred_weighted = np.sum(SVRs_preds_weighted, axis=0)
            return SVRs_pred_weighted