#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph-based molecular regression

Created on Tue Aug 28 19:47:10 2018

@author: remussn


1st try:
    - use kernel provided by GraKel(    )
    - use SVM regressor to do regression

"""
#%% 0. Molecules
'''
1. 乙酸乙酯 ethyl_acetate
CH₃COOCH₂CH₃
    H3C-C-O-CH2-CH3
      ||
      O
2. 四氢呋喃 oxolane
    C-C-C-C
    |     |
    ---O---
3. N,N-二甲基甲酰胺 DMF
    H3C--N--CH==O
         |
         CH3
4. 丙酮 Propanone
    CH3-C-CH3
        ||
        O
5. 噻吩 Thiophene
    CH=CH-CH=CH
    |        |
    ----S-----
6. 甲苯 Toluenes
    ⌬-CH3
7. 1,4-二氧六环 dioxane_1_4
    O- CH2 -CH2
    |       |
    CH2-CH2-O
8. 甲醇 Methanol
    CH3OH
9. 氯苯 PhCl
    ⌬-Cl
10. 仲丁醇 sec_butanol
    CH3-Ch2-CH-Ch3
            |
            OH
11. 二苯甲酮 Benzophenone
    ⌬-C-⌬
      ||
      O
12. 苯胺 aniline
    ⌬-NH2
13. 糠醛 furfural
               H
               |
    CH=CH-CH=C-C=O
    |        |
    ----O-----
14. 氯化苄 Benzyl Chloride
    ⌬-CH2-cl, BnCl
15. 环己酮 Cyclohexanone
    CH2-CH2-CH2-CH2-CH2
    |               |
    -------C--------
           ||
           O
16. s

1   2   3   4   5
C - C - C - C - C
            |
            C 5'
            |
            C 6'

'''
#%% 1. Parameters
bond_type_to_weight_dict = {'single':1, 'double': 2, 'pi': 3}

#%% 2. build graphs
from molecule import Molecule
# m0: H2O
m0 = Molecule(
        name = 'H2O',
        name_Chinese='水',
        bone_atoms_list = {'O':['1a']},
        side_atoms_list = {'H':['1a2']})

'''
1. 乙酸乙酯 ethyl_acetate
      1a  2a  3a  4a    5a
 H3 - C - C - O - CH2 - CH3
          ||
          O
          3b
5%          
'''
#ethyl_acetate_01 = Molecule(atoms_list=['H-3','C','O','C','H-2','C','H-3'], 
#                            bonds_list=[[1,4,'single'],[2,4,'single'],[3,4,'single'],
#                                        [4,5,'double'],[4,6,'single'],
#                                        [6,7,'single'],[]]])
# example: CH3CH3
m1 = Molecule(
        name = 'ethyl_acetate',
        name_Chinese='乙酸乙酯',
        bone_atoms_list = {'C':['1a','2a','4a','5a'],
                           'O':['3a','3b']},
        side_atoms_list = {'H':['1a3','4a2','5a3']},
        additional_or_special_bonds_list = [['2a','3b','double']])
y1 = 5
'''
2. 四氢呋喃 oxolane
    C-C-C-C
    |     |
    ---O---
4%
'''
m2 = Molecule(
        name = 'oxolane',
        name_Chinese='四氢呋喃',
        bone_atoms_list={'C':['1a','2a','3a','4a'],
                     'O':['5a']},
        side_atoms_list={'H':['1a2','2a2','3a2','4a2']},
        additional_or_special_bonds_list=[['5a','1a']])
y2 = 4
'''
3. N,N-二甲基甲酰胺 DMF
    H3C--N--CH==O
         |
         CH3
'''
m3 = Molecule(
        name = 'DMF',
        name_Chinese='N,N-二甲基甲酰胺',
        bone_atoms_list={'C':['1a','3a','3b'],
                     'O':['4a'],
                     'N':['2a']},
        side_atoms_list={'H':['1a3','3a','3b3']},
        additional_or_special_bonds_list=[['3a','4a','double']])
y3 = 5
'''
4. 丙酮 Propanone
    CH3-C-CH3
        ||
        O
'''
m4 = Molecule(
        name = 'Propanone',
        name_Chinese='丙酮',
        bone_atoms_list={'C':['1a','2a','3a'],
                     'O':['3b']},
        side_atoms_list={'H':['1a3','3a3']},
        additional_or_special_bonds_list=[['2a','3b','double']])
y4 = 5
'''
5. 噻吩 Thiophene
    CH=CH-CH=CH
    |        |
    ----S-----
    
    CH-CH-CH-CH
    |  (PI)   |
    ----S-----
'''
m5_1 = Molecule(
        name = 'Thiophene',
        name_Chinese='噻吩',
        bone_atoms_list={'C':['1a','2a','3a','4a'],
                     'S':['5a']},
        side_atoms_list={'H':['1a','2a','3a','4a']},
        additional_or_special_bonds_list=[['1a','2a','double'],['3a','4a','double'],['5a','1a','single']])
m5_2 = Molecule(
        name = 'Thiophene',
        name_Chinese='噻吩',
        bone_atoms_list={'C':['1a','2a','3a','4a'],
                     'S':['5a']},
        side_atoms_list={'H':['1a','2a','3a','4a']},
        additional_or_special_bonds_list=[['1a','2a','pi'],
                                          ['2a','3a','pi'],
                                          ['3a','4a','pi'],
                                          ['2a','3a','pi'],
                                          ['4a','5a','pi'],
                                          ['5a','1a','pi']])
y5 = 4
'''
6. 甲苯 Toluenes
    ⌬-CH3
'''
m6 = Molecule(
        name = 'Toluenes',
        name_Chinese='甲苯',
        bone_atoms_list={'Ph':['1a'],
                         'C':['2a']},
        side_atoms_list={'H':['2a3']})
y6 = 4
'''
7. 1,4-二氧六环 dioxane_1_4
    1a
    O- CH2 -CH2
    |       |
    CH2-CH2-O
'''
m7 = Molecule(
        name = 'dioxane_1_4',
        name_Chinese='1,4-二氧六环',
        bone_atoms_list={'C':['2a','3a','5a','6a'],
                         'O':['1a','4a']},
        side_atoms_list={'H':['2a2','3a2','5a2','6a2']},
        additional_or_special_bonds_list=[['6a','1a','single']])
y7 = 2
'''
8. 甲醇 Methanol
    CH3OH
'''
m8 = Molecule(
        name = 'Methanol',
        name_Chinese='甲醇',
        bone_atoms_list={'C':['1a'],'O':['2a']},
        side_atoms_list={'H':['1a3','2a']})
y8 = 5
'''
9. 氯苯 PhCl
    ⌬-Cl
'''
m9 = Molecule(
        name = 'PhCl',
        name_Chinese='氯苯',
        bone_atoms_list={'Ph':['1a'], 'Cl': ['2a']})
y9 = 6
'''
10. 仲丁醇 sec_butanol
    CH3-Ch2-CH-Ch3
            |
            OH
'''
m10 = Molecule(
        name = 'Sec Butanol',
        name_Chinese='仲丁醇',
        bone_atoms_list={'C':['1a', '2a','3a','4a'],
                         'O':['4b']},
        side_atoms_list={'H':['1a3','2a2','3a','4a3','4b']}
        )
y10 = 2
'''            
11. 二苯甲酮 Benzophenone
    ⌬-C-⌬
      ||
      O
'''
m11 = Molecule(
        name = 'Benzophenone',
        name_Chinese='二苯甲酮',
        bone_atoms_list={'Ph':['1a','3a'],
                         'C':['2a'],
                         'O':['3b']},
        additional_or_special_bonds_list=[['2a','3b','double']])
y11 = 4
'''
12. 苯胺 aniline
    ⌬-NH2
'''
m12 = Molecule(
        name = 'Aniline',
        name_Chinese='苯胺',
        bone_atoms_list={'Ph':['1a'],
                         'N':['2a']},
        side_atoms_list={'H':['2a2']})
y12 = 7
'''
13. 糠醛 furfural
                   H
               4a  |
    CH=CH-CH = C - C = O
    |          |   5a  6a
    ----O-------
        5b
'''
m13_1 = Molecule(
        name = 'furfural',
        name_Chinese='糠醛',
        bone_atoms_list={'C':['1a', '2a','3a','4a', '5a'],
                         'O':['6a','5b']},
        side_atoms_list={'H':['1a','2a','3a','5a']},
        additional_or_special_bonds_list=[['1a','2a','double'],
                                          ['3a','4a','double'],
                                          ['5a','6a','double'],
                                          ['5b','1a','single']])
m13_2 = Molecule(
        name = 'furfural',
        name_Chinese='糠醛',
        bone_atoms_list={'C':['1a', '2a','3a','4a', '5a'],
                         'O':['6a','5b']},
        side_atoms_list={'H':['1a','2a','3a','5a']},
        additional_or_special_bonds_list=[['1a','2a','pi'],
                                          ['2a','3a','pi'],
                                          ['3a','4a','pi'],
                                          ['4a','5b','pi'],                                          
                                          ['5b','1a','pi'],
                                          ['5a','6a','double']])
y13 = 98
'''
14. 氯化苄 Benzyl Chloride
    ⌬-CH2-cl, BnCl
'''
m14 = Molecule(
        name = 'Benzyl Chloride',
        name_Chinese='氯化苄',
        bone_atoms_list={'Ph':['1a'],
                         'C':['2a'],
                         'Cl':['3a']},
        side_atoms_list={'H':['2a2']})
y14 = 96
'''
15. 环己酮 Cyclohexanone
    CH2-CH2-CH2-CH2-CH2
    |               |
    -------C--------
           ||
           O
'''
m15 = Molecule(
        name = 'Cyclohexanone',
        name_Chinese='环己酮',
        bone_atoms_list={'C':['1a', '2a','3a','4a', '5a', '6a'],
                         'O':['7b']},
        side_atoms_list={'H':['1a2','2a2','3a2','4a2','5a2']},
        additional_or_special_bonds_list=[['6a','1a','single'],
                                          ['6a','7b','double']])
y15 = 95
'''
16. 2-噻吩甲醛 2-Thenaldehyde
1a   2a  3a 4a 5a
CH - S - CH-CH=O
|        |  
CH   -   CH
5b       4b
'''
m16_2 = Molecule(
        name = '2-Thenaldehyde',
        name_Chinese='2-噻吩甲醛',
        bone_atoms_list={'C':['1a','3a','4a','5a','4b','5b'],
                         'S':['2a'],
                         'O':['5a']},
        side_atoms_list={'H':['1a','3a','4a','4b','5b']},
        additional_or_special_bonds_list=[['1a','2a','pi'],
                                          ['2a','3a','pi'],
                                          ['3a','4b','pi'],
                                          ['4b','5b','pi'],
                                          ['5b','1a','pi'],
                                          ['4a','5a','double']])
y16 = 95
'''
17. 苯甲酸
Ph - C = O
     |
     OH
'''
m17 = Molecule(
        name = 'Benzoic acid',
        name_Chinese='苯甲酸',
        bone_atoms_list={'Ph':['1a'],
                         'C':['2a'],
                         'O':['3a','3b']},
        side_atoms_list={'H':['3b']},
        additional_or_special_bonds_list=[['2a','3a','double']])
y17 = 92
'''
18. 二甲基亚砜 Dimethyl sulfoxide
CH3 - S - CH3
      ||
      O
'''
m18 = Molecule(
        name = 'Dimethyl sulfoxide',
        name_Chinese='二甲基亚砜',
        bone_atoms_list={'C':['1a','3a'],
                         'S':['2a'],
                         'O':['3b']},
        side_atoms_list={'H':['1a3','3a3']},
        additional_or_special_bonds_list=[['2a','3b','double']])
y18 = 40
'''
19. 苄醇(苯甲醇)Benzyl alcohol
Ph-CH2-OH
'''
m19 = Molecule(
        name = 'Benzyl alcohol',
        name_Chinese='苯甲醇',
        bone_atoms_list={'Ph':['1a'],
                         'C':['2a'],
                         'O':['3a']},
        side_atoms_list={'H':['2a2','3a']})
y19 = 30
'''
20. 苯甲醛 Benzaldehyde
Ph-CH=O
'''
m20 = Molecule(
        name = 'Benzaldehyde',
        name_Chinese='苯甲醛',
        bone_atoms_list={'Ph':['1a'],
                         'C':['2a'],
                         'O':['3a']},
        side_atoms_list={'H':['2a']},
        additional_or_special_bonds_list=[['2a','3a','double']])
y20 = 98

#%%
from kernelSVR import kernelSVR
ks = kernelSVR()
gk = GraphKernel(kernel={"name": "multiscale_laplacian",
                             "which": "fast",
                             "L": 1,
                             "P": 10,
                             "N": 10})
#ks.add_kernel(gk)
ignoreH = False
expandPh = True
mx_use = toGraKelList(mx_train, ignoreH, expandPh)
#mx_use = toGraKelList(mx_full, ignoreH, expandPh)
ks.fit_kernel(mx_use)#, my_train)
ks.fit_SVRs(my_train)

mx_use_test = toGraKelList(mx_test, ignoreH, expandPh)
#mx_use_test = mx_use
ks_pred_all = ks.predict(mx_use_test, 'all')
ks_pred = ks.predict(mx_use_test)


#%%
from grakel import GraphKernel, datasets
mutag = datasets.fetch_dataset("MUTAG", verbose=False)
mutag_data = mutag.data     # list of 188 graphs
wl_kernel = GraphKernel(kernel = [{"name": "weisfeiler_lehman", "niter": 5}, {"name": "subtree_wl"}], normalize=True)
split_point = int(len(mutag_data) * 0.9)
X_train, X_test = mutag_data[:split_point], mutag_data[split_point:]
K_train = wl_kernel.fit_transform(X_train)
K_test = wl_kernel.transform(X_test)
y = mutag.target
y_train, y_test = y[:split_point], y[split_point:]
from sklearn.svm import SVC
clf = SVC(kernel='precomputed')
clf.fit(K_train, y_train)
y_pred = clf.predict(K_test)
from sklearn.metrics import accuracy_score
print("%2.2f %%" %(round(accuracy_score(y_test, y_pred)*100)))

#%%
import numpy as np
from grakel import GraphKernel
#from prettytable import PrettyTable
ignoreH = True
expandPh = True
def toGraKelList(molecule_list, ignoreH = False, expandPh = False):
    if expandPh:
        molecule_list = [m.expandPh() for m in molecule_list]
    if ignoreH:
        molecule_list = [m.ignoreH() for m in molecule_list]
    return [m.to_GraKel_graph() for m in molecule_list]

def print_name_pred_real_dif(names, pred, real):
    dif = pred-real
    #x = PrettyTable()
    #x.field_names = ["Name", "Pred", "Real", "Diff"]
    
    for idx in range(len(names)):
    #    x.add_row([names[idx], str(pred[idx]), str(real[idx]), str(dif[idx])])
    #print(x)
        if len(names[idx]) >6:
            print(names[idx]+'\t'+str(pred[idx])+'\t\t'+str(real[idx])+'\t'+str(dif[idx]))
        else:    
            print(names[idx]+'\t\t'+str(pred[idx])+'\t\t'+str(real[idx])+'\t'+str(dif[idx]))

#def kernelTable(kernel, molecule_names):
#    import pandas as pd
#    kernel_tb = pd.DataFrame(K_full, index = [str(idx+1)+'-'+m.name_Chinese for [idx,m] in enumerate(mx_raw)], 
#                                              columns = [str(idx+1)+'-'+m.name_Chinese for [idx,m] in enumerate(mx_raw)])

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

train_idx = np.subtract([1,2,3,4,5,8,9,10,11,12,13,14,16,17,18,19,20],1)
test_idx = np.subtract([6,7,15,18],1)
mx_full = [m1,m2,m3,m4,m5_2,m6,m7,m8,m9,m10,m11,m12,m13_2,m14,m15,m16_2,m17,m18,m19,m20]
my_full = [y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20]
mx_train = [mx_full[idx] for idx in train_idx]; my_train = [my_full[idx] for idx in train_idx]
mx_test = [mx_full[idx] for idx in test_idx];   my_test  = [my_full[idx] for idx in test_idx]

testKernel(mx_train=mx_train, my_train = my_train, mx_test=mx_test, my_test=my_test)
EhKernel = GraphKernel(kernel = [{"name": "edge_histogram"}], normalize=True)
testKernel(mx_train=mx_train, my_train = my_train, mx_test=mx_test, my_test=my_test, kernel = EhKernel)
# KEY:
# RING + DOUBLE O BOUND


#mx = toGraKelList([m1,m2,m3,m4,m5_2,m6,m7,m8,m9,m10,m11,m12,m13_2,m14,m15,m16_2,m17,m18,m19,m20], ignoreH, expandPh)
#my = [y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20]

#mx_train = toGraKelList([m1,m2,m3,m4,m5_2,m8,m9,m10,m11,m12,m13_2,m14,m16_2,m17,m19,m20], ignoreH, expandPh)
#mx_test = toGraKelList([m1, m6, m7, m15, m18], ignoreH, expandPh)
#my_train = [y1,y2,y3,y4,y5,y8,y9,y10,y11,y12,y13,y14,y16,y17,y19,y20]
#my_test = [y1, y6, y7, y15, y18]

#from grakel import GraphKernel
#wl_kernel = GraphKernel(kernel = [{"name": "weisfeiler_lehman", "niter": 20}, {"name": "subtree_wl"}], normalize=False)
##wl_kernel_full = GraphKernel(kernel = [{"name": "weisfeiler_lehman", "niter": 2}, {"name": "subtree_wl"}], normalize=False)
##K_full = wl_kernel_full.fit_transform(mx)
#K_train = wl_kernel.fit_transform(mx_train)
#K_test = wl_kernel.transform(mx_test)
#
#from sklearn.svm import SVR
##reg_full = SVR(kernel='precomputed')
##reg_full.fit(K_full, my)
#reg = SVR(kernel='precomputed')
#reg.fit(K_train, my_train)
## my_pred = reg.predict(K_test)
##print([int(pred) for pred in reg_full.predict(K_full)])
##print(my)
#
#print([int(pred) for pred in reg.predict(K_train)])
#print(my_train)
#print(reg.predict(K_test))
#print(my_test)



#%%
from grakel import GraphKernel
import scipy
wl_kernel = GraphKernel(kernel=[{"name": "weisfeiler_lehman"}, {"name": "subtree_wl"}], normalize = True)
sp_kernel = GraphKernel(kernel={"name": "shortest_path"}, normalize=True)
m0_edges, m0_labels = m0.to_GraKel_graph()
m1_edges, m1_labels = m1.to_GraKel_graph()
H2O = [[[[0, 1, 1], [1, 0, 0], [1, 0, 0]], {0: 'O', 1: 'H', 2: 'H'}]]
# H2O = scipy.sparse.csr_matrix(([1, 1, 1, 1], ([0, 0, 1, 2], [1, 2, 0, 0])), shape=(3, 3))
sp_kernel.fit_transform(H2O)
# sp_kernel.fit_transform([m1_edges,m1_labels])
# sp_kernel.fit_transform([[m0_edges.todense().tolist(),m0_labels]])
sp_kernel.fit_transform([[m0_edges,m0_labels]])