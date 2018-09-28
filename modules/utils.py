#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 20:42:03 2018

@author: remussn
"""
import random
import numpy as np
from grakel import GraphKernel
def train_test_split(xs, ys, train_frac=0.7):    
    shuffle_idx = list(range(len(xs)))
    random.shuffle(shuffle_idx)
    
    train_num = int(len(xs)*train_frac)
    train_idx = shuffle_idx[:train_num]
    m_train = [xs[idx] for idx in train_idx]
    y_train = [ys[idx] for idx in train_idx]
    
    test_idx = shuffle_idx[train_num:]
    m_test = [xs[idx] for idx in test_idx]
    y_test = [ys[idx] for idx in test_idx]
    
    return m_train, y_train, m_test, y_test
    

def toGraKelList(molecule_list, ignoreH = False, expandPh = False):
    if expandPh:
        molecule_list = [m.expandPh() for m in molecule_list]
    if ignoreH:
        molecule_list = [m.ignoreH() for m in molecule_list]
    return [m.to_GraKel_graph() for m in molecule_list]


def print_name_pred_real_dif(names, pred, real):
    dif = pred-real    
    print('名称\t\t预测值\t\t真实值\t差')
    for idx in range(len(names)):
        if len(names[idx]) >6:
            print(names[idx]+'\t'+str('%.4f'%pred[idx])+'\t\t'+str(real[idx])+'\t'+str('%.4f'%dif[idx]))
        else:    
            print(names[idx]+'\t\t'+str('%.4f'%pred[idx])+'\t\t'+str(real[idx])+'\t'+str('%.4f'%dif[idx]))

def testKernel(mx_train, my_train,mx_test, my_test,
               kernel = GraphKernel(kernel = [{"name": "weisfeiler_lehman", "niter": 10}, {"name": "subtree_wl"}], normalize=False)):
    # wl_kernel = GraphKernel(kernel = [{"name": "weisfeiler_lehman", "niter": 10}, {"name": "subtree_wl"}], normalize=False)
    K_train = kernel.fit_transform(toGraKelList(mx_train))
    K_test = kernel.transform(toGraKelList(mx_test))
    
    from sklearn.svm import SVR
    reg = SVR(kernel='precomputed')
    reg.fit(K_train, my_train[:16])    
    
    print('TRAIN')
    print_name_pred_real_dif([m.name_Chinese for m in mx_train], reg.predict(K_train), my_train)
    print('TEST')
    print_name_pred_real_dif([m.name_Chinese for m in mx_test], reg.predict(K_test), my_test)