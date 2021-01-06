# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 20:40:46 2020

@author: heiba
"""

import os
os.chdir('D:\GithubRepo\RGCN-LinkPrediction')
import argparse
import numpy as np
import time
import torch
from torch.utils.data import DataLoader
import random
from dgl.contrib.data import load_data
from pprint import pprint
from pathlib import Path
import pandas as pd

from utils import build_graph, node_norm_2_edge_norm, get_adj, generate_sampled_graph_and_labels, preprocess
from model import LinkPredict
from dataset import TestDataset


def get_model():
    return LinkPredict(num_nodes=29416,
                       h_dim=50,
                       num_rels=2,
                       num_bases=2,
                       num_hidden_layers=12,
                       dropout=0.2,
                       reg_param=0.01)


def load_model(path):
    state = torch.load(path)
    model.load_state_dict(state['state_dict'])
    #optimizer.load_state_dict(state['optimizer'])


def calc_score(output, triplets):
    sub = output[triplets[:, 0]]  # [triple num, dim]
    obj = output[triplets[:, 2]]  # [triple num, dim]
    w_relation = torch.nn.Parameter(torch.Tensor(2, 50))
    r = w_relation[triplets[:, 1]]  # [triple num, dim]
    # DistMult: sub.T@diag(r)@obj
    score = torch.sum(sub * r * obj, dim=1)  # [triple num]
    return score

# =============================================================================
# def evaluate(split='valid'):
#     model.eval()
#     with torch.no_grad():
#         output = model(graph, test_node_id, rel, test_edge_norm)  # [ent_num, dim]
#     print(output)
#     mrr, hits_dict = self.calc_mrr(output, split, hits=[1, 3, 10], filtered=self.p.filtered)
#     return mrr, hits_dict
# =============================================================================

candidatelink=np.array(pd.read_csv('candidatetripe.tsv', sep ="\t", header=None))
tsdata=np.array(pd.read_csv('drkg_train_nw.tsv', sep ="\t", header=None))
train_data=tsdata
# =============================================================================
# tsdata=np.array(pd.read_csv('drkg_train_nw.tsv', sep ="\t", header=None))
# validata=np.array(pd.read_csv('drkg_valid_nw.tsv', sep ="\t", header=None))
# testdata=np.array(pd.read_csv('drkg_test_nw.tsv', sep ="\t", header=None))
# #有多少个节点，也就是节点有多少种类。
# =============================================================================
num_nodes=29416
#edge的种类
num_rels=2
num_nodes=29416,
n_hidden=50,
num_rels=2,
n_bases=2,
n_layers=12,
dropout=0.2,
regularization=0.01
#self.num_nodes, self.train_data, self.valid_data, self.test_data, self.num_rels = numofnodes, tsdata, validata, testdata, numofrels
device = torch.device('cpu')
graph, rel, node_norm = build_graph(num_nodes=29416, num_rels=2,edges=train_data)        
model = get_model()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
test_node_id = torch.arange(0, 29416, dtype=torch.long).view(-1, 1).to(device)
test_edge_norm = node_norm_2_edge_norm(graph, torch.from_numpy(node_norm).view(-1, 1)).to(device)
        



save_path='D:/GithubRepo/RGCN-LinkPrediction/checkpoints/2020_11_10_08_10_02.pt'
load_model(save_path)
model.eval()
with torch.no_grad():
    output = model(graph, test_node_id, rel, test_edge_norm)  # [ent_num, dim]
    print(output)

triplet=torch.from_numpy(candidatelink)
scores = calc_score(output=output, triplets=triplet)


scores.tolist()
len(scores.tolist())#551072
scorelist=scores.tolist()

with open("scores.csv", 'w+') as f:
    for x in scorelist:
        f.writelines("{}\n".format(x[0]))

listofcandidate=candidatelink.tolist()
linkscore=[]
for x in range(0,len(scorelist)):
    linkscore.append([listofcandidate[x],scorelist[x]])
    

forcal=[[x] for x in scorelist]
linkscoreser=pd.Series(forcal,index=listofcandidate)

linkscore[:64]
benchmark1=linkscore[64][1]
filterdlinkscore1=[]
for x in linkscore:
    if x[1]>benchmark1:
        filterdlinkscore1.append(x)

filterdlinkscore1[:100]
benchmark2=filterdlinkscore1[99][1]
filterdlinkscore2=[]
for x in linkscore:
    if x[1]>benchmark2:
        filterdlinkscore2.append(x)

filterdlinkscore2[:100]
benchmark3=filterdlinkscore2[98][1]
filterdlinkscore3=[]
for x in linkscore:
    if x[1]>benchmark3:
        filterdlinkscore3.append(x)


filterdlinkscore3[:100]
benchmark4=filterdlinkscore3[2][1]
filterdlinkscore4=[]
for x in linkscore:
    if x[1]>benchmark4:
        filterdlinkscore4.append(x)

###sort up.

candidrugid=[]
for x in filterdlinkscore4:
    candidrugid.append(x[0][0])

len(candidrugid)

with open("candidrugid.csv", 'w+') as f:
    for x in candidrugid:
        f.writelines("{}\n".format(x))




