#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Molecule Structure Class

Created on Wed Aug 29 18:53:34 2018

@author: remussn
"""
import re
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class Molecule(nx.Graph):
    def __init__(self, name = 'unamed_molecule',
                 name_Chinese = '未命名分子',
                 bone_atoms_dict = None, leaf_atoms_dict = None,
                 default_bond_type = 'single', 
                 additional_or_special_bonds_list = None,
                 bond_type_to_weight_dict = {'single':1, 'double': 5, 'pi': 10},
                 atom_type_to_label_dict = {'C': 1, 'H': 2, 'O': 3, 'S': 4, 'N': 5, 'Cl': 6, 'Ph': 7}):
        """Initalize molecule structure        
        Args:
            name: 
                name of molecule
                分子英文名
            name_Chinese:
                Chinsese name of molecule
                分子中文名
            bone_atoms_dict:
                Atoms on main chains, should be dict of atom:[locations]: {'C':['1a','2a'}
                主链原子，格式应为 atom:[locations] 形式的字典，例如： {'C':['1a','2a'}
            leaf_atoms_list:
                "Side" atoms
                
        """
        nx.Graph.__init__(self)
        self.name = name
        self.name_Chinese = name_Chinese
        self.default_bond_type = default_bond_type
        self.atom_type_to_label_dict = atom_type_to_label_dict
        self.bond_type_to_weight_dict = bond_type_to_weight_dict
        
        if bone_atoms_dict != None:
            self.add_bone_atoms(bone_atoms_dict)
        if leaf_atoms_dict != None:
            self.add_leaf_atoms(leaf_atoms_dict)
        if additional_or_special_bonds_list != None:
            self.add_bonds(additional_or_special_bonds_list)
                    
    
    def get_adj_mat(self):
        return nx.adj_matrix(self, weight = 'bond_weight')
    
    
    def add_bone_atoms(self, atoms_dict):
        """Add bone atoms
        Args:
            atoms_dict: {'C':['1a','2a','3a','4a','5a','6a','5b','6b'],
                         'O':['6c']}
        
        """
        for atom in atoms_dict:
            for atom_location in atoms_dict[atom]:
                # Add this atom to molecule
                loc_index = int(re.findall('\d',atom_location)[0])      # '5' in '5b'                
                chain_index = re.findall('[a-z]',atom_location)[0]      # 'b' in '5b'
                self.add_node(atom_location, atom = atom, atom_label = self.atom_type_to_label_dict[atom], leaf_atom_num = 0)# ('5b','C',1)
                
                # Add bonds in same branch/chain
                prev_loc_index = loc_index - 1                  # 5 - 1 = 4
                next_loc_index = loc_index + 1
                if self.has_node(str(prev_loc_index)+chain_index):
                    # self.add_edge(atom_location, str(prev_loc_index)+chain_index, bond_type = 'single')
                    self.add_bonds([[atom_location, str(prev_loc_index)+chain_index]])
                if self.has_node(str(next_loc_index)+chain_index):
                    self.add_bonds([[atom_location, str(next_loc_index)+chain_index]])
                
                
                # Add bonds to neighbor atoms in another branch/chain
                # prev('5b') = ['4b','4a']
                prev_chain_index = chr(ord(chain_index)-1)      # b - 1 = a
                prev_location = str(prev_loc_index) + prev_chain_index
                if self.has_node(prev_location):
                    self.add_bonds([[atom_location, prev_location]])
                                
                next_chain_index = chr(ord(chain_index)+1)
                next_location = str(next_loc_index) + next_chain_index
                if self.has_node(next_location):
                    self.add_bonds([[atom_location, next_location]])
                    
                
            
    def add_leaf_atoms(self, atoms_dict):
        """Add side atoms
        Args:
            atoms_dict: {'H':['1a3',['2a2','single'],'3a2','4a','5a2','6a3','6b3']
                         }
        
        """
        for atom in atoms_dict:
            for atom_location_num_bond in atoms_dict[atom]:
                # Extract atom location+number and bond type(if have)
                if type(atom_location_num_bond) == str:
                    atom_location_num = atom_location_num_bond
                    bond_type = None
                else:
                    atom_location_num, bond_type = atom_location_num_bond
                    
                # Extract atom location and number
                # can be '1a' or '1a3'
                chain_index = re.findall('[a-z]',atom_location_num)[0]            
                loc_index, number = atom_location_num.split(chain_index)
                number = 1 if number == '' else int(number)
                
                # Add atom and bond to molecule
                atom_location = loc_index + chain_index
                for atom_idx in range(number):
                    leaf_atom_location = atom_location + '+'*(self.nodes[atom_location]['leaf_atom_num']+1)
                    self.add_node(leaf_atom_location, atom = atom, atom_label = self.atom_type_to_label_dict[atom])
                    self.nodes[atom_location]['leaf_atom_num'] += 1
                    self.add_bonds([[atom_location, leaf_atom_location]])
                    
                    
                # [loc_index, number] = [int(item) for item in atom_location.split(chain_index)]    
    
    def add_bonds(self, bonds_locs_type_list):
        """
        Add bond by location such as ['1a', '2a']
        Args:
            bonds_locs_type_list: list of bonds locations and types(if no, set as default), [['1a', '2a'],['2a', '3a','double']]
            bonds_type_list: list of bonds types, ['single','single']
        
        """
                
        for bond_idx, bond_locs_type in enumerate(bonds_locs_type_list):
            if len(bond_locs_type) ==2:
                bond_type = self.default_bond_type
            else:
                bond_type = bond_locs_type[2]
            self.add_edge(bond_locs_type[0], bond_locs_type[1], bond_type = bond_type, bond_weight = self.bond_type_to_weight_dict[bond_type])    
        
    def expandPh(self):
        """ 
        Expand Ph to C-cycle, new atom will be placed on new chains, of which chain index 'z','y',...        
        """
        import copy
        moleclue_expandPh = copy.deepcopy(self)
        init_chain_idx = 'z'
        # chr(ord(init_chain_idx) - 1)
        Ph_locs = [node for node in moleclue_expandPh.nodes if moleclue_expandPh.nodes[node]['atom']=='Ph']
        if len(Ph_locs) >= 1:
            for Ph_idx, Ph_loc in enumerate(Ph_locs):
                # Get loc of neighs of Ph
                Ph_neighbors_locs = list(moleclue_expandPh.neighbors(Ph_loc))
                # Remove (old) Ph from molecule
                moleclue_expandPh.remove_node(Ph_loc)
                # Add new atoms
                chain_idx = chr(ord(init_chain_idx) - Ph_idx)
                atoms_locs = [str(i+1)+chain_idx for i in range(6)]
                for c_idx in range(6):
                    # Adding Atoms
                    moleclue_expandPh.add_node(atoms_locs[c_idx], atom = 'C', 
                                               atom_label = self.atom_type_to_label_dict['C'], leaf_atom_num = 0)
                for c_idx in range(6):
                    # Adding bonds
                    moleclue_expandPh.add_bonds([[atoms_locs[c_idx], atoms_locs[(c_idx+1)%6],'pi']])
                    
                if len(Ph_neighbors_locs) == 1:
                    # Re-link Ph's neighbors
                    moleclue_expandPh.add_bonds([[atoms_locs[0], Ph_neighbors_locs[0], 'single']])
                    # Add H atoms
                    moleclue_expandPh.add_leaf_atoms({'H':atoms_locs[1:]})
                else:
                    print('don\'t support multineighbors!')    
                
                return moleclue_expandPh
        else:
            return self
    
    def ignoreH(self):
        """ Remove all H from molecule
        """
        import copy
        Hs_list = [node for node in self.nodes if self.nodes[node]['atom']=='H']
        moleclue_ignoreH = copy.deepcopy(self)
        moleclue_ignoreH.remove_nodes_from(Hs_list)
        return moleclue_ignoreH
        # return self.copy(), Hs_list
        
    
    def print_atoms(self):
        for atom in self.nodes.data():
            print(atom)

    def print_bonds(self, mode = 'seperate'):
        # modes: 'seperate', 'bone only', 'colp'
        bond_type_str_dict = {'single':'-',
                              'double': '=',
                              'pi': '~'}
        for bond in self.edges.data():
            atom1_loc = bond[0]; atom1 = self.nodes[atom1_loc]['atom']
            atom2_loc = bond[1]; atom2 = self.nodes[atom2_loc]['atom']
            bond_type = bond[2]['bond_type']; bond_str = bond_type_str_dict[bond_type]
            
            # 1a-C -- 2a-C | single
            # print('{}-{} {} {}-{}'.format(atom1_loc, atom1, bond_str, atom2, atom2_loc))
            
            # (1a) C-C (2a)
            print('({}) {} {} {} ({})'.format(atom1_loc, atom1, bond_str, atom2, atom2_loc))
            
    def visualize(self):
        '''
        https://python-graph-gallery.com/325-map-colour-to-the-edges-of-a-network/
        https://python-graph-gallery.com/324-map-a-color-to-network-nodes/
        '''
#        bond_type_color_dict = {'single':'-',
#                              'double': '=',
#                              'pi': '~'}
        start_atoms = []
        end_atoms = []
        edge_weights = []        
        for bond in self.edges.data():
            atom1_loc = bond[0]; atom1 = self.nodes[atom1_loc]['atom']; 
            start_atoms.append(atom1+'-'+atom1_loc)
            atom2_loc = bond[1]; atom2 = self.nodes[atom2_loc]['atom']; 
            end_atoms.append(atom2+'-'+atom2_loc)
            bond_weight = bond[2]['bond_weight'];   edge_weights.append(bond_weight)
            
            #bond_type = bond[2]['bond_type'];   edge_weights.append(bond_type)
        # Dataframe with connections
        df = pd.DataFrame({ 'from':start_atoms, 'to':end_atoms, 'value':edge_weights})        
        G=nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.Graph() )
        
        nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_color=df['value'], width=10.0, edge_cmap=plt.cm.Blues)         
        
    def to_GraKel_graph(self):
        # H2O = scipy.sparse.csr_matrix(([1, 1, 1, 1], ([0, 0, 1, 2], [1, 2, 0, 0])), shape=(3, 3))
        edges = self.get_adj_mat()
        
        # H2O atom labels = {0: 'O', 1: 'H', 2:'H'}
        indices = list(range(len(self.nodes)))
        atoms = [self.nodes[atom]['atom'] for atom in self.nodes]
        atoms_labels = dict(zip(indices, atoms))
        
        # H2O bond labels = {0: 'single', 1: 'single', 2:'single'}
        bond_weights = [self.edges[bond]['bond_weight'] for bond in self.edges]
        bond_labels_dict = dict(zip(indices, bond_weights))
        return edges, atoms_labels, bond_labels_dict
        