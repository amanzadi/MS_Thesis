# ================ Imorting Packages =====================
import os
import dgl
import torch as th
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import glob


# =========== Making drugs specific graphs ===============

# Node file
nodes = pd.read_csv('../data/GNN-GSE_full_pkd_norm.csv', sep=',')

# Edge file
edges = pd.read_csv('../data/GNN-PPI-net.csv', sep=',')

# Drug-protein file (DTI)
dti = pd.read_csv('../data/GNN-DTI_full.csv', sep=',')

# DrugIds
DrugID = pd.read_csv('../data/DrugID.csv', sep = ',')

# ======= Creating a Dataset for Graph Classification from CSV =======
'''
Creating a Dataset for Graph Classification from CSV
1. graph_edges.csv
containing three columns:

graph_id: the ID of the graph.
src: the source node of an edge of the given graph.
dst: the destination node of an edge of the given graph.
'''

# generate drug specific ppi subgraph for GNN edges
def drug2ppi(drug):
    genes = dti['ProteinID'].loc[dti['DrugID'] == drug].tolist()
    df = edges[['protein1','protein2']].loc[edges['protein1'].isin(genes)]
    df = df.loc[df['protein2'].isin(genes)]
    df['graph_id'] = DrugID.loc[DrugID['DrugID'] == drug]['GraphID'].tolist()[0]  #DrugID
    df = df.rename(columns={'protein1': 'src_prot', 'protein2': 'dst_prot'}) # prot: actual protein id
    final_genes =df['src_prot'].unique().tolist() # final genes that have ppi data
    dic = {gene:final_genes.index(gene) for gene in final_genes} # conversion dic, starts at 0
    df['src'] = df['src_prot'].map(dic) #local ids
    df['dst'] = df['dst_prot'].map(dic) #local ids
    return(df[['graph_id', 'src', 'dst', 'src_prot', 'dst_prot']])

#Edges of the garphs, graph_edges.csv
GNN_edges = pd.DataFrame(columns=['graph_id', 'src', 'dst', 'src_prot', 'dst_prot'])

print('Calculating edges ...')

for drug in tqdm(dti['DrugID'].unique().tolist()):
    path = '../data/GNN_edges/'+ str(drug)+'.csv'
    drug2ppi(drug).to_csv(path, index=False, sep = ',')

print('edges are calculated, concatenating files ...')

os.chdir('../data/GNN_edges')
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in tqdm(all_filenames) ])

#export to feather
combined_csv.reset_index().to_feather('GNN_edges.feather')

print('GNN_edges.feather is now created and saved!')


os.chdir('../Code')
print('GNN_edges.feather is now created and saved!')


'''
2.graph_properties.csv: containing three columns:
graph_id: the ID of the graph. -label: the label of the graph. -num_nodes: the number of nodes in the grap
'''

# counts how many nodes a drug2ppi has
def drug2num_nodes(drug):
    genes = dti['ProteinID'].loc[dti['DrugID'] == drug].tolist()
    df = edges[['protein1','protein2']].loc[edges['protein1'].isin(genes)]
    df = df.loc[df['protein2'].isin(genes)]
    return(len(df['protein1'].unique()))


print('Calculating properties ...')
#labes of the garphs, graph_properties.csv
rows = []
for i,drug in tqdm(enumerate(dti['DrugID'].unique().tolist())):
    row = {'graph_id': i+1,
           'label': drug, #drugID
           'num_nodes': drug2num_nodes(drug)}
    rows.append(row)
    
GNN_properties = pd.DataFrame.from_dict(rows)   
GNN_properties.to_csv('../data/GNN_properties.csv', index=False, sep = ',')

print('GNN_properties.csv is now created and saved!')
print('\n============================================\n')
print('GNN graphs preparation is done !!!')
