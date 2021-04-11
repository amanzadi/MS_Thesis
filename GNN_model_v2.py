# ============================ 1. Environment setup ============================
import os
import dgl
import time
import torch
import pickle
import numpy as np
import pandas as pd
import networkx as nx
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

        drug_list = drug_comb['Drug1'].unique().tolist()
        for drug in drug_comb['Drug2'].unique().tolist():
            if drug not in drug_list:
                drug_list.append(drug)
        graphs_list = properties.loc[properties['label'].isin(drug_list)]['graph_id'].unique().tolist()

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
        for graph_id in tqdm(graphs_list):

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
        drug2graph = {self.labels[i]:i for i in range(len(self.labels))} 

        print('\nMaking the combinational graphs and their labels (PSE)\n')
        for i in tqdm(range(len(drug_comb))):
            row = drug_comb.loc[i]
            g1 = self.graphs[drug2graph[row[0]]] # Drug1 graph
            g2 = self.graphs[drug2graph[row[1]]] # Drug2 graph  
            self.comb_graphs.append([g1,g2])
            self.comb_labels.append(torch.tensor(row[2:])) # PSE values

            
    def __getitem__(self, i):
       # return self.comb_graphs[i], self.comb_labels[i]
        return self.comb_graphs[i], self.comb_labels[i]

    def __len__(self):
        return len(self.comb_graphs)
    
print('\nCreating the PolypharmacyDataset ...\n')

dataset = PolypharmacyDataset()

print('\nPolypharmacyDataset created!\n')

end = time.time()
hours, rem = divmod(end-data_time , 3600)
minutes, seconds = divmod(rem, 60)

print('\ndataset is compiled! \ncompiling time = {:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(minutes),seconds))

# ========================= 3. Data loading and batch =========================

print('\nCreating train and test batches ...\n')

from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

num_examples = len(dataset)
num_train = int(num_examples * 0.8)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = GraphDataLoader(
    dataset, sampler=train_sampler, batch_size=35 ,drop_last=False)
test_dataloader = GraphDataLoader(
    dataset, sampler=test_sampler, batch_size=35 ,drop_last=False)

print('\nTrain and test batches are created!\n')

# ========================= 4. GNN Model: Siamese GCN =========================

from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats,  num_classes)
        
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        out = F.relu(dgl.mean_nodes(g, 'h'))
        #out = F.relu(dgl.max_nodes(g, 'h'))
        return out

# =============================== 5. Training =================================
train_time = time.time()

#print('\nCreating the SiameseGCN model ...\n')

# Create the model with given dimensions
model = GCN(dataset.dim_nfeats, 200, dataset.gclasses)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print('\n======================== Trainig ========================\n')

f = open("log_model_V2.txt", "a")
f.write('\n======================== Trainig ========================\n')

for epoch in range(50):
    batch = 0
    for batched_graph, labels in train_dataloader:
        start_time = time.time()
        g1 = batched_graph[0]
        g2 = batched_graph[1]
        pred1 = model(g1, g1.ndata['PSE'].float())
        pred2 = model(g2, g2.ndata['PSE'].float())
        pred = F.normalize(pred1+pred2)/2
        loss = F.binary_cross_entropy(torch.sigmoid(pred).float(),labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        run_time = round(time.time() - start_time,3)
        msg = 'epoch%s, batch%s | loss = %s | time: %s s\n' % (epoch,batch,loss.tolist(),run_time)
        f.write(msg)
        print (msg)
        batch += 1

end = time.time()
hours, rem = divmod(end-train_time , 3600)
minutes, seconds = divmod(rem, 60)

msg_tiem = '\ntotal training time = {:0>2}:{:0>2}:{:05.2f}'.format(int(hours),int(minutes),seconds)
f.write(msg_tiem)
print(msg_tiem)

torch.save(model.state_dict(), 'state_dict_model_V2.pt')
torch.save(model, 'entire_model_V2.pt')
print('\nmodel is saved!\n')
f.write('\nmodel saved!\n')

print('\n======================== Testing ========================\n')

f.write('\n======================== Testing ========================\n')

test_time = time.time()

TP = 0
FP = 0
TN = 0
FN = 0
for batched_graph, labels in test_dataloader:
    pred1 = model(batched_graph[0], batched_graph[0].ndata['PSE'].float())
    pred2 = model(batched_graph[1], batched_graph[1].ndata['PSE'].float())
    pred = F.normalize(pred1+pred2)/2

    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] == 1 and pred[i][j] != 0:
                TP += 1
            elif labels[i][j] == 1 and pred[i][j] == 0:
                FN += 1
            elif labels[i][j] == 0 and pred[i][j] != 0:
                FP += 1
            elif labels[i][j] == 0 and pred[i][j] == 0:
                TN += 1
                pass
            
    # Validation metrics        
    acc = ((TP+TN)*100)/(TP+FP+FN+TN)
    prec = (TP*100)/(TP+FP)
    recall = (TP*100)/(TP+FN)
    F1 = 2*(recall*prec)/(recall+prec)
    sim = ((F.cosine_similarity(pred.float(),labels.float())).mean().tolist())*100
    msg2 = 'Accuracy: %s | Precision: %s | Recall: %s | F1: %s | Similarity: %s\n' %(round(acc,4),round(prec,4),round(recall,4),round(F1,4),round(sim,4))
    f.write(msg2)
    print(msg2)


end = time.time()
hours, rem = divmod(end-test_time , 3600)
minutes, seconds = divmod(rem, 60)
msg_time2 = '\ntotal testing time = {:0>2}:{:0>2}:{:05.2f}\n'.format(int(hours),int(minutes),seconds)
f.write(msg_time2)
print(msg_time2)

f.write('Testing is done!')
print('\nTesting is done!\n')



f.close()

 