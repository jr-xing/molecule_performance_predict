#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 20:36:26 2018

@author: remussn
"""

#%% 1. Load Data
import numpy as np
from modules.molecule_dataset import molecule_structures, molecule_performance
from modules.utils import train_test_split
m_train, y_train, m_test, y_test = train_test_split(molecule_structures, molecule_performance, 0.8)

# 2. 
from modules.utils import toGraKelList, print_name_pred_real_dif
from modules.kernelSVR import KernelSVR
ignoreH = False
expandPh = True
ks = KernelSVR()
ks.fit(toGraKelList(m_train, ignoreH, expandPh), y_train)
#ks.fit_kernel(toGraKelList(m_train, ignoreH, expandPh))
ks_pred_train = ks.predict(toGraKelList(m_train, ignoreH, expandPh))
ks_pred_train_all = ks.predict(toGraKelList(m_train, ignoreH, expandPh), 'all')
ks_pred_test = ks.predict(toGraKelList(m_test, ignoreH, expandPh))
print_name_pred_real_dif([m.name_Chinese for m in m_train], ks_pred_train, y_train)
print_name_pred_real_dif([m.name_Chinese for m in m_test], ks_pred_test, y_test)