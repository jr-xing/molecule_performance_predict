#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 20:37:25 2018

@author: remussn
"""

from .molecule import Molecule
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

molecule_structures = [m1,m2,m3,m4,m5_2,m6,m7,m8,m9,m10,m11,m12,m13_2,m14,m15,m16_2,m17,m18,m19,m20]
molecule_performance = [y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19,y20]