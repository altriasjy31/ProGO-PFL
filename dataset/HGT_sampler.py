import dgl
import dgl.data
import dgl.function as fn
import numpy as np
import torch as th
import scipy.sparse as ssp
import array
import torch



# class Budget(object):
#     def __init__(self, hg, n_types, NS):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.n_types = {}
#         for key, value in n_types.items():
#             self.n_types[key] = th.zeros(value).to(device)
#         self.NS = NS
#         self.hg = hg
#     def update(self, dst_type, idxs):
#         for etype in self.hg.canonical_etypes:
#             if dst_type == etype[2]:
#                 src_type = etype[0]
#                 #degree = self.hg.in_degrees(idx, etype=etype)
#                 for i in idxs:
#                     src_idx = self.hg.predecessors(i, etype=etype)
#                     #if src_idx.shape[0] > 0:
#                     len = src_idx.shape[0]
#                     if src_type in self.NS.keys():
#                         src_idx = th.tensor([i for i in src_idx if i not in self.NS[src_type]])
#                     if src_idx.shape[0] > 0:
#                         self.n_types[src_type][src_idx] += 1 / len

#     def pop(self, type, idx):
#         self.n_types[type][idx] = 0


# class HGTsampler(dgl.dataloading.Sampler):
#     def __init__(self, hg, category, num_nodes_per_type, num_steps):
#         self.n_types = {}
#         for n in hg.ntypes:
#             self.n_types[n] = hg.num_nodes(n)
#         self.category = category
#         self.num_nodes_per_type = num_nodes_per_type
#         self.num_steps = num_steps
#         self.hg = hg

#     def sampler_subgraph(self, seed_nodes):
#         # OS = {self.category: th.stack(seed_nodes)}
#         OS = seed_nodes
#         NS = OS
#         B = Budget(self.hg, self.n_types, NS)
#         for type, idxs in OS.items():
#             B.update(type, idxs)
#         for i in range(self.num_steps):
#             prob = {}
#             for src_type, p in B.n_types.items():
#                 #print(src_type)
#                 if p.max() > 0:
#                     prob[src_type] = p / th.sum(p)
#                     sampled_idx = th.multinomial(prob[src_type], self.num_nodes_per_type, replacement=False)
#                     if not OS.__contains__(src_type):
#                         OS[src_type] = sampled_idx
#                     else:
#                         OS[src_type] = th.cat((OS[src_type], sampled_idx))
#                     B.update(src_type, sampled_idx)
#                     B.pop(src_type, sampled_idx)
#         sg = self.hg.subgraph(OS)
#         return sg, OS
    
#     def sample(self, g, seeds):
#         sg, OS = self.sampler_subgraph(seeds)
        
#         return sg.ndata[dgl.NID], seeds, sg



import dgl
from dgl.dataloading import Sampler
import numpy as np
import torch

class HGTsampler(Sampler):
    def __init__(self, g, num_sampling, device):
        self.num_sampling = num_sampling
        self.device = device
        
    def sample(self, g, seeds):
        # for etype in g.etypes:
        #     g.edges[etype].data['edge_prob'] = torch.ones(g.number_of_edges(etype)).to(self.device) * self.prob
        # subgraph = dgl.sampling.sample_neighbors(g, seeds, -1, prob='edge_prob')
        # all_edges = {}
        # for etype in subgraph.canonical_etypes:
        #     src, dst = subgraph.edges(etype=etype)
        #     all_edges[etype] = (src, dst)
        # subgraph_nodes = {}
        # for etype in all_edges:
        #     src, dst = all_edges[etype]
        #     subgraph_nodes[etype] = torch.cat([src, dst]).unique()

        # subgraph = dgl.edge_subgraph(g, subgraph_nodes)
        
        # node_dict = {}
        # if 'go_annotation' not in seeds.keys():
        #     seeds['go_annotation'] = torch.tensor([], dtype=torch.int64).to(self.device)
        # for ntype in subgraph.ntypes:
        #     node_dict[ntype] = torch.unique(torch.cat((subgraph.ndata['_ID'][ntype], seeds[ntype])))
            
        # subgraph = dgl.node_subgraph(g, node_dict)
        
        if 'go_annotation' not in seeds.keys():
            seeds['go_annotation'] = torch.tensor([], dtype=torch.int64).to(self.device)
        frontier = dgl.sampling.sample_neighbors(g, seeds, -1)
        edges_dict = {}
        # for etype in frontier.etypes:
        #     edges_dict[etype] = frontier.edata['_ID'][etype]
        subg_no_isolated = dgl.edge_subgraph(g, frontier.edata['_ID'], relabel_nodes=True, store_ids=True)
        original_node_ids = subg_no_isolated.ndata[dgl.NID]
        # subg = dgl.node_subgraph(g, seeds)
        # original_node_ids = subg.nodes['user'].data[dgl.NID]
        num_1 = original_node_ids['protein'].shape[0]
        num_2 = original_node_ids['go_annotation'].shape[0]
        if num_2 > 0:
            x1 = int(self.num_sampling * (num_1 / (num_1+num_2)))
            x2 = self.num_sampling - x1
            rand_idx1 = torch.randint(0, num_1, size=(x1,))
            rand_idx2 = torch.randint(0, num_2, size=(x2,))
        
            selected_protein = original_node_ids['protein'][rand_idx1]
            selected_go_annotation = original_node_ids['go_annotation'][rand_idx2]
            selected_protein = torch.unique(torch.cat((selected_protein, seeds['protein'])))
            selected_go_annotation = torch.unique(torch.cat((selected_go_annotation, seeds['go_annotation'])))
            selected_nodes_dict = {'protein':selected_protein, 'go_annotation':selected_go_annotation}
            subgraph = dgl.node_subgraph(g, selected_nodes_dict)
        else:
            selected_protein = original_node_ids['protein']
            selected_protein = torch.unique(torch.cat((selected_protein, seeds['protein'])))
            selected_nodes_dict = {'protein':selected_protein}
            subgraph = dgl.node_subgraph(g, selected_nodes_dict)
        
        return selected_nodes_dict, seeds, subgraph