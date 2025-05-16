import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import dgl
import numpy as np
import pandas as pd
import scipy.sparse as ssp
import random
import pathlib as P
from collections import Counter, defaultdict
import pickle
from pathlib import Path

# define the dataset for link prediction

class DBLPDataset(Dataset):
    def __init__(self, dataset_name, go_name, model_name, batch_size,  device):
        self.device = device
        self.data_path = P.Path(__file__).absolute().parent.parent.joinpath('{}/{}'.format(dataset_name, go_name))
        self.result_path = P.Path(__file__).absolute().parent.parent.joinpath('results/{}_{}.txt'.format(model_name, go_name))
        self.data_output_path = P.Path('./{}/{}'.format(dataset_name, go_name))
        self.g_path = self.data_path.joinpath('graph.dgl')
        self.train_g_path = self.data_path.joinpath('inductive_train_graph.dgl')
        self.test_g_path = self.data_path.joinpath('inductive_test_graph.dgl')
        
        self.g = self.create_graph().to(device)
        self.train_idx, self.valid_idx, self.test_idx = self.get_split('default')
        self.train_idx = self.train_idx.to(self.device)
        self.valid_idx = self.valid_idx.to(self.device)
        self.test_idx = self.test_idx.to(self.device)
        
        self.del_valid_test()
        self.train_g_nodes_dict = {'protein':torch.cat((self.train_idx, self.test_idx)), 'go_annotation':range(self.g.number_of_nodes('go_annotation'))}
        self.train_g = dgl.node_subgraph(self.g, self.train_g_nodes_dict)
        self.test_g = self.get_test_g(k=2)
        
        
        self.feature_dim = len(self.g.ndata['h']['protein'][0])
        self.protein_num = self.g.number_of_nodes('protein')
        self.go_num = self.g.number_of_nodes('go_annotation')
        # self.input_feature = HeteroLinear.HeteroFeature({'protein': h_dict['protein']}, get_nodes_dict(self.g), embed_size=128, act=None)
        self.label_dict = self.load_label_dict(self.data_path.joinpath('label.pkl'))
        self.node_type = ['protein', 'go_annotation']
        self.category = 'protein'
        # protein_protein_sample_size = 5 
        # protein_go_sample_size = 3     
        # go_go_sample_size = 2            

        
        # 创建自定义采样器
        # self.sampler = hg_Sampler(self.g, protein_protein_sample_size, protein_go_sample_size, go_go_sample_size)
        # self.sampler = HANSampler(self.g, seed_ntypes=[self.category], meta_paths_dict=self.meta_paths_dict, num_neighbors=5)
        # self.sampler = dgl.dataloading.MultiLayerNeighborSampler([3,3])
        self.sampler = dgl.dataloading.NeighborSampler([5])

        self.batch_size = batch_size
        self.train_loader = dgl.dataloading.DataLoader(
                    self.train_g, {self.category: self.train_idx.to(self.device), 'go_annotation':torch.arange(0, self.go_num)}, self.sampler,
                    batch_size=self.batch_size, device=device, shuffle=True)
        self.valid_loader = dgl.dataloading.DataLoader(
                    self.train_g, {self.category: self.valid_idx.to(self.device)}, self.sampler,
                    batch_size=self.batch_size, device=device, shuffle=True)
        self.test_loader = dgl.dataloading.DataLoader(
                    self.test_g, {self.category: self.test_idx.to(self.device)}, self.sampler,
                    batch_size=self.test_idx.shape[0], device=device, shuffle=True)
        # self.test_types = list(self.links_test['data'].keys()) if edge_types == [] else edge_types
        # self.features_list = self.get_features_list()
        # self.adjM = sum(self.links['data'].values())
        # self.g = self.create_graph()
        # self.e_feat = self.get_e_feats()
 
    def del_valid_test(self):
        etype_to_remove_1 = ('protein', 'interacts_0', 'go_annotation')
        etype_to_remove_2 = ('go_annotation', '_interacts_0', 'protein')
        self.g = dgl.remove_edges(self.g, self.valid_idx, etype=etype_to_remove_1)
        self.g = dgl.remove_edges(self.g, self.valid_idx, etype=etype_to_remove_2)
        self.g = dgl.remove_edges(self.g, self.test_idx, etype=etype_to_remove_1)
        self.g = dgl.remove_edges(self.g, self.test_idx, etype=etype_to_remove_2)
        
    
    def create_graph(self):
        data_path = self.data_path
        g_path = P.Path(self.g_path)
        if os.path.exists(g_path):
            g, _ = dgl.load_graphs(str(g_path))
            return g[0]
        else:
            node = pd.read_csv(data_path.joinpath('node.dat'), header=None, sep='\t')
            node.columns = ['id', 'name', 'type', 'feature']
            node['id'] = node.groupby('type').cumcount()
            node.loc[node['type'] == 1, 'feature'] = None
            output_path = str(data_path.joinpath('node.dat'))  # 目标文件路径
            node.to_csv(output_path, sep='\t', index=False, header=False)
            feature_dict = {'protein':[], 'go_annotation':[]}
            for _, row in node.iterrows():
                    if row['type'] == 0:
                        feature_dict['protein'].append(row['feature'].split(','))
                    if row['type'] == 1:
                        feature_dict['go_annotation'].append(row['feature'])
            protein_features = [list(map(float, feature)) for feature in feature_dict['protein']]
            protein_tensor = th.tensor(protein_features)
            emb_size = len(feature_dict['protein'][0])
            go_num = node[node['type'] == 1].shape[0]
            go_indices = torch.randint(0, go_num, (go_num,))
            feature_dict = {'protein': protein_tensor, 'go_annotation': F.one_hot(go_indices, num_classes=go_num)}

            protein_num = len(feature_dict['protein'])
            link = pd.read_csv(data_path.joinpath('link.dat'), header=None, sep='\t')
            link_test = pd.read_csv(self.data_path.joinpath('link.dat.test'), header=None, sep='\t')
            link = pd.concat([link, link_test], axis=0)           
            p2g_src = link[link[2]==0][0].values
            p2g_des = link[link[2]==0][1].values
            p2p_src = link[link[2]==1][0].values
            p2p_des = link[link[2]==1][1].values
            g2g_src = link[link[2]==2][0].values
            g2g_des = link[link[2]==2][1].values
            p2g_des = [go-protein_num for go in p2g_des]
            g2g_src = [go-protein_num for go in g2g_src]
            g2g_des = [go-protein_num for go in g2g_des]
            
            # edge_attrs_tensor = th.tensor(edge_attrs, dtype=th.float32)
            graph_data = {
                ('protein', 'interacts_0', 'go_annotation'): (torch.tensor(p2g_src),torch.tensor(p2g_des)),
                ('go_annotation', '_interacts_0', 'protein'): (torch.tensor(p2g_des),torch.tensor(p2g_src)),
                ('protein', 'interacts_1', 'protein'): (torch.tensor(p2p_src),torch.tensor(p2p_des)),
                ('protein', '_interacts_1', 'protein'): (torch.tensor(p2p_des),torch.tensor(p2p_src)),
                ('go_annotation', 'interacts_2', 'go_annotation'): (torch.tensor(g2g_src),torch.tensor(g2g_des)),
                ('go_annotation', '_interacts_2', 'go_annotation'): (torch.tensor(g2g_des),torch.tensor(g2g_src))
            }
            g = dgl.heterograph(graph_data)

            
            g.ndata['h'] = feature_dict
            for ntype in g.ntypes:
                g.nodes[ntype].data[dgl.NID] = torch.arange(g.num_nodes(ntype))
            # g = dgl.remove_self_loop(g)
            # g = dgl.add_self_loop(g)

            dgl.save_graphs(str(self.data_path.joinpath('graph.dgl')), g)
            return g

        # g_path = self.data_path.joinpath('graph.dgl')
        # if os.path.exists(g_path):
        #     g, _ = dgl.load_graphs(g_path)
        #     return g
        # else:
        #     g = dgl.DGLGraph(self.adjM+(self.adjM.T))
        #     g = dgl.remove_self_loop(g)
        #     g = dgl.add_self_loop(g)
        #     dgl.save_graphs(self.data_path.joinpath('graph.dgl'), g)
        #     return g
    
    def get_split(self, mode='default', validation=True):
        r"""
        
        Parameters
        ----------
        validation : bool
            Whether to split dataset. Default ``True``. If it is False, val_idx will be same with train_idx.

        We can get idx of train, validation and test through it.

        return
        -------
        train_idx, val_idx, test_idx : torch.Tensor, torch.Tensor, torch.Tensor
        """
        if mode == 'default':
            link_test = pd.read_csv(self.data_path.joinpath('link.dat.test'), header=None, sep='\t')
            test_idx = link_test.iloc[:, 0].drop_duplicates().values.tolist()
            all_nodes = list(range(self.g.number_of_nodes('protein')))
            train_idx = [node for node in all_nodes if node not in test_idx]
            train_idx = torch.tensor(train_idx, dtype=torch.long)  # 转为 tensor 类型
            test_idx = torch.tensor(test_idx, dtype=torch.long)  

        elif mode == 'random':
            num_nodes = self.g.number_of_nodes(self.category)
            n_test = int(num_nodes * 0.2)
            n_train = num_nodes - n_test

            train, test = torch.utils.data.random_split(range(num_nodes), [n_train, n_test])
            train_idx = torch.tensor(train.indices)
            test_idx = torch.tensor(test.indices)
        if validation:
            random_int = torch.randperm(len(train_idx))
            valid_idx = train_idx[random_int[:len(train_idx) // 5]]
            train_idx = train_idx[random_int[len(train_idx) // 5:]]
        else:
            valid_idx = train_idx
            train_idx = train_idx

        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx
        # Here set test_idx as attribute of dataset to save results of HGB
        return self.train_idx, self.valid_idx, self.test_idx

    def load_label_dict(self, file_path):
        # 提取link和link_test中蛋白质-go的前两列，pid-gid
        label_path = self.data_path.joinpath('label.pkl')
        if os.path.exists(label_path):
            with open(file_path, 'rb') as f:
                label_dict = pickle.load(f)
        else:
            node = pd.read_csv(self.data_path.joinpath('node.dat'), header=None, sep='\t')
            link = pd.read_csv(self.data_path.joinpath('link.dat'), header=None, sep='\t')  
            link = link[link[2]==0]
            link_test = pd.read_csv(self.data_path.joinpath('link.dat.test'), header=None, sep='\t')
            protein_num = node[node[2]==0].shape[0]
            link[1] = link[1] - protein_num
            link_test[1] = link_test[1] - protein_num
            src = list(link[0]) + list(link_test[0])
            des = list(link[1]) + list(link_test[1])

            num = len(src)
            protein_num = self.protein_num
            go_num = node[node[2]==1].shape[0]

            label_protein = list(link[0]) + list(link_test[0])
            label_go = list(link[1]) + list(link_test[1])

            label_dict = {'protein':label_protein, 
                        'go':label_go}
            output_file = open(label_path, 'wb')
            pickle.dump(label_dict, output_file)

        # 将 label_dict['protein'] 的值与索引构建成一个字典，加快查找速度
        protein_to_go = {}
        for i, protein in enumerate(label_dict['protein']):
            if protein not in protein_to_go:
                protein_to_go[protein] = []
            protein_to_go[protein].append(label_dict['go'][i])

        return protein_to_go # 返回最大 go 编号作为类别数

    def get_label(self, protein_idx):
        # 初始化一个 (len(protein_idx), go_num) 的 label_list
        label_list = [[0] * self.go_num for _ in range(len(protein_idx))]
        
        # 加速查找，避免每次都访问 dict
        label_dict = self.label_dict
        
        # 映射 protein_idx 到 label_list 的索引，减少重复查找
        protein_idx_map = {idx: i for i, idx in enumerate(protein_idx)}

        # 遍历 label_dict 中的内容
        for idx in protein_idx:
            if idx in label_dict:
                for go in label_dict[idx]:
                    label_list[protein_idx_map[idx]][go] = 1  # 使用 protein_idx_map 映射加快速度

        return label_list
    
    def save_results(self, pred, label):
        node = pd.read_csv(self.data_path.joinpath('node.dat'), header=None, sep='\t')
        node.columns = ['id', 'name', 'type', 'feature']
        protein_node = node[node['type']==0]
        protein_name = protein_node['name']
        
        pred = pred.numpy()
        label = label.numpy()
        test_idx = list(self.test_idx.cpu())
        protein_name = protein_name[test_idx]
        
        np.savez(self.result_path, name=protein_name, labels=label, preds=pred)
        
    def get_test_g(self, k=1):
        # new_node_id = list(self.test_idx.cpu())
        # current_nodes = {'protein':new_node_id}
        # for i in range(k):
        #     current_subgraph = dgl.out_subgraph(self.g, current_nodes, relabel_nodes=True)
        #     current_nodes = {}
        #     for ntype in self.g.ntypes:
        #         node_ids = current_subgraph.nodes(ntype).tolist()
        #         current_nodes[ntype] = node_ids
        # return current_subgraph
        test_idx = list(self.test_idx.cpu())
        test_g = dgl.node_subgraph(self.g, {'protein':test_idx})
        return test_g