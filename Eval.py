# ============================ 1. Environment setup ============================
import os
import dgl
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from sklearn import metrics
import torch.nn.functional as F
from dgl.data import DGLDataset

# ========================= 2. Setting up the Dataset =========================

class PSE_eval(DGLDataset):
    def __init__(self):
        super().__init__(name='PSE_eval')

    def process(self):
        features = pd.read_csv('../data/GNN-GSE_full_pkd_norm.csv',index_col = 'ProteinID', sep=',')
        drug_comb = pd.read_csv('../data/Eval_TWOSIDE-evaluation-PSE-964.csv', sep=',') 
        nodes = pd.read_csv('../data/GNN-GSE_full_pkd_norm.csv', sep=',')
        edges = pd.read_csv('../data/GNN-PPI-net.csv', sep=',')
        dti = pd.read_csv('../data/Eval_DTI_full.csv', sep=',')
        DrugID = pd.read_csv('../data/Eval_DrugID.csv', sep = ',')
        print('data loaded!')
        
        # generate drug specific ppi subgraph for GNN edges
        def drug2ppi(drug):
            genes = dti['ProteinID'].loc[dti['DrugID'] == drug].tolist()
            df = edges[['protein1','protein2']].loc[edges['protein1'].isin(genes)]
            df = df.loc[df['protein2'].isin(genes)]
            num_nodes = len(df['protein1'].unique())
            df['graph_id'] = DrugID.loc[DrugID['DrugID'] == drug]['GraphID'].tolist()[0]  #DrugID
            df = df.rename(columns={'protein1': 'src_prot', 'protein2': 'dst_prot'}) # prot: actual protein id
            final_genes =df['src_prot'].unique().tolist() # final genes that have ppi data
            dic = {gene:final_genes.index(gene) for gene in final_genes} # conversion dic, starts at 0
            df['src'] = df['src_prot'].map(dic) #local ids
            df['dst'] = df['dst_prot'].map(dic) #local ids
            return(df[['graph_id', 'src', 'dst', 'src_prot', 'dst_prot']],num_nodes)
        
        self.graphs = []
        self.labels = []
        self.comb_graphs = []
        self.comb_labels = []

        #Node features or PSEs dictionary
        feature_dic = {i+1:torch.tensor(features.loc[i+1,]) for i in range(len(features))}
    
        # For each graph ID...
        for drug in tqdm(DrugID['DrugID'].tolist()[:20]):
            # Find the edges as well as the number of nodes and its label.
            edges_of_id,num_nodes = drug2ppi(drug)
            src = edges_of_id['src'].to_numpy()
            dst = edges_of_id['dst'].to_numpy()
            label = DrugID.loc[DrugID['DrugID'] == drug]['Name'].tolist()[0]
            
            # Create a graph and add it to the list of graphs and labels.
            g = dgl.graph((src, dst), num_nodes=num_nodes)
            
            # Need to convert proteinsIDs for feature assigning
            prot_ids = edges_of_id['src_prot'].unique().tolist()
            for prot in edges_of_id['dst_prot'].unique().tolist():
                if prot not in prot_ids:
                    prot_ids.append(prot)
            convert_prot = {prot_ids.index(prot):prot for prot in prot_ids}
            
            #Adding features of each node
            g.ndata['PSE'] = torch.zeros(g.num_nodes(), 964)
            for node in g.nodes().tolist():
                g.ndata['PSE'][node] = feature_dic[convert_prot[node]]
                
            self.graphs.append(g)
            self.labels.append(label)
            
        # conver drugid to their respective graph id
        #drug2graph = {properties['label'][i]:i for i in range(len(properties))} 
        #drug2graph = {self.labels[i]:i for i in range(len(self.labels))} 
        
        for i in range(len(drug_comb)):
            row = drug_comb.loc[i]
            try:
                g1 = self.graphs[self.labels.index(row[0])] # Drug1 graph
                g2 = self.graphs[self.labels.index(row[1])] # Drug2 graph  
                self.comb_graphs.append([g1,g2])
                self.comb_labels.append(torch.tensor(row[2:])) # PSE values
            except:
                pass

            

    def __getitem__(self, i):
        return self.comb_graphs[i], self.comb_labels[i]
        #return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.comb_graphs)
    
print('\nCreating the Evaluation Dataset ...\n')

dataset = PSE_eval()

print('\nEvaluation Dataset created!\n')

print('\ndataset is compiled! \n')

# ========================= 3. Data loading and batch =========================

print('\nCreating eval batches ...\n')
# Making the batches
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

num_examples = len(dataset)
eval_sampler = SubsetRandomSampler(torch.arange(num_examples))
eval_dataloader = GraphDataLoader(dataset, sampler=eval_sampler, batch_size=1, drop_last=False)

print('\nEval batches are created!\n')

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

# =============================== 5. Evaluation =================================


#print('\nInitilizing pretrained the SiameseGCN model ...\n')

# Specify a path
PATH = "entire_model_V2.pt"
conf = 'state_dict_model_V2.pt'

# Load
model = GCN(964,200,964)
model.load_state_dict(torch.load(conf))
model.eval()
print('\n======================== Evaluating ========================\n')

f = open("Eval.txt", "a")
f.write('\n======================== Evaluating ========================\n')

def predict(g1, g2):  # graph1, graph2
    pred1 = model(g1, g1.ndata['PSE'].float())
    pred2 = model(g2, g2.ndata['PSE'].float())
    pred = F.normalize(pred1+pred2)/2
    return(pred)



for batched_graph, labels in eval_dataloader:
    g1 = batched_graph[0]
    g2 = batched_graph[1]
    pred = predict(g1, g2)
    print(pred)
    f.write(pred)
    print(labels)
    f.write(labels)

    TP = 0
    FP = 0
    TN = 0
    FN = 0
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
            else:
                pass
            
    # Validation metrics        
    acc = ((TP+TN)*100)/(TP+FP+FN+TN)
    prec = (TP*100)/(TP+FP)
    recall = (TP*100)/(TP+FN)
    F1 = 2*(recall*prec)/(recall+prec)
    sim = ((F.cosine_similarity(pred.float(),labels.float())).mean().tolist())*100
    x = [TP,TN,FP,FN]
    f.write('TP:%s,TN:%s,FP:%s,FN:%s' % (x[0],x[1],x[2],x[3]))
    print('TP:%s,TN:%s,FP:%s,FN:%s' % (x[0],x[1],x[2],x[3]))
    msg2 = 'Accuracy: %s | Precision: %s | Recall: %s | F1: %s | Similarity: %s\n' %(round(acc,4),round(prec,4),round(recall,4),round(F1,4),round(sim,4))
    f.write(msg2)
    print(msg2)


f.close()

