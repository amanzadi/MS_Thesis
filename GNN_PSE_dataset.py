# ============================ 1. Environment setup ============================
import os
import dgl
import time
import torch
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from dgl.data import DGLDataset

# ========================= 2. Setting up the Dataset =========================

data_time = time.time()

class PolypharmacyDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='polypharmacy')

    def process(self):
        print('\nloading data ...\n')
        edges = pd.read_feather('../data/GNN_edges.feather')
        #edges = pd.read_csv('../data/GNN_edges-toy.csv')
        properties = pd.read_csv('../data/GNN_properties.csv')
        drug_comb = pd.read_csv('../data/GNN-TWOSIDE-train-PSE-964.csv', sep=',') # or 3347
        features = pd.read_csv('../data/GNN-GSE_full_pkd_norm.csv', index_col = 'ProteinID', sep=',')

        self.graphs = []
        self.labels = []
        self.comb_graphs = []
        self.comb_labels = []
        
        num_features = len(features.columns) # no. of PSEs
        self.dim_nfeats = num_features
        self.gclasses = num_features

        print('\nNumber of polypharmacy side effects (PSE):',num_features)
        print('\nInitilizing parameters...\n')

        # Create a graph for each graph ID from the edges table.
        # First process the properties table into two dictionaries with graph IDs as keys.
        # The label and number of nodes are values.
        label_dict = {}
        num_nodes_dict = {}
        
        for _, row in properties.iterrows():
            label_dict[row['graph_id']] = row['label']
            num_nodes_dict[row['graph_id']] = row['num_nodes']

        # For the edges, first group the table by graph IDs.
        #edges_group = edges.groupby('graph_id')
        
        #Node features or PSEs dictionary
        feature_dic = {i+1:torch.tensor(features.loc[i+1,]) for i in range(len(features))}
        

        print('\nMaking the individual graphs and their features (PSE)\n')

        # For each graph ID...
        for graph_id in tqdm(properties['graph_id'].tolist()):

            # Find the edges as well as the number of nodes and its label.
            edges_of_id = edges.loc[edges['graph_id'] == graph_id]
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            num_nodes = num_nodes_dict[graph_id]
            label = label_dict[graph_id]
            
            # Create a graph and add it to the list of graphs and labels.
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            
            # Need to convert proteinsIDs for feature assigning
            prot_ids = edges_of_id['src_prot'].unique().tolist()
            for prot in edges_of_id['dst_prot'].unique().tolist():
                if prot not in prot_ids:
                    prot_ids.append(prot)
            convert_prot = {prot_ids.index(prot):prot for prot in prot_ids}
            
            #Adding features of each node
            g.ndata['PSE'] = torch.zeros(g.num_nodes(), num_features)
            for node in g.nodes().tolist():
                g.ndata['PSE'][node] = feature_dic[convert_prot[node]]
                
            self.graphs.append(g)
            self.labels.append(label)
        
        # conver drugid to their respective graph id
        #drug2graph = {properties['label'][i]:i for i in range(len(properties))} 
        drug2graph = {self.labels[i]:i for i in range(len(self.labels))} 

        print('\nMaking the combinational graphs and their labels (PSE)\n')
        for i in tqdm(range(len(drug_comb))):
            row = drug_comb.loc[i]
            g1 = self.graphs[drug2graph[row[0]]] # Drug1 graph
            g2 = self.graphs[drug2graph[row[1]]] # Drug2 graph  
            self.comb_graphs.append([g1,g2])
            self.comb_labels.append(torch.tensor(row[2:])) # PSE values

            
        # Convert the label list to tensor for saving.
        #self.comb_labels = torch.LongTensor(self.comb_labels)

    def __getitem__(self, i):
       # return self.comb_graphs[i], self.comb_labels[i]
        return self.comb_graphs[i], self.comb_labels[i]

    def __len__(self):
        return len(self.comb_graphs)
    
print('\nCreating the PolypharmacyDataset ...\n')

dataset = PolypharmacyDataset()

file = open('PolypharmacyDataset.obj', 'w') 
pickle.dump(dataset, file)

print('\nPolypharmacyDataset.obj created!\n')

end = time.time()
hours, rem = divmod(end-data_time , 3600)
minutes, seconds = divmod(rem, 60)

print('\ndataset is compiled! \ncompiling time = {:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(minutes),seconds))
