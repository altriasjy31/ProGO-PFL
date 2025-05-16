import torch
import dgl
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import random
import pathlib as P
import os
import pickle


class GCN_Dataset(Dataset):
    def __init__(self, dataset_name, go_name, batch_size, device):
        self.data_path = P.Path(__file__).absolute().parent.parent.joinpath('{}/{}'.format(dataset_name, go_name))
        self.result_path = P.Path(__file__).absolute().parent.parent.joinpath('results/gcn_{}.txt'.format(go_name))
        hg, _ = dgl.load_graphs(str(self.data_path.joinpath('graph.dgl')))
        self.hg = hg[0]
        self.ppi_path = str(self.data_path.joinpath('PPI.dgl'))
        self.ppi = self.create_graph().to(device)
        # ppi = dgl.to_homogeneous(ppi, ndata = 'h')
        self.device = device
        # self.feature_dim = len(self.ppi.ndata['h'][0])
        # self.protein_num = self.ppi.number_of_nodes()
        self.feature_dim = self.hg.ndata['h']['protein'].shape[1]
        self.protein_num = self.hg.ndata['h']['protein'].shape[0]
        self.go_num = self.get_go_num()
        self.label_dict = self.load_label_dict(self.data_path.joinpath('label.pkl'))
        
        # go_feature = torch.ones((self.go_num, self.feature_dim), dtype=torch.float32)
        # feature_dict = hg.ndata['h']
        # feature_dict['go_annotation'] = go_feature
        # hg.ndata['h'] = feature_dict
        
        
        self.train_idx, self.valid_idx, self.test_idx = self.get_split()
        
        # self.ppi = self.create_full_ppi().to(device)
        
        # etype_to_remove_1 = ('protein', 'interacts_0', 'go_annotation')
        # etype_to_remove_2 = ('go_annotation', '_interacts_0', 'protein')
        # hg = dgl.remove_edges(hg, self.valid_idx, etype=etype_to_remove_1)
        # hg = dgl.remove_edges(hg, self.valid_idx, etype=etype_to_remove_2)
        # hg = dgl.remove_edges(hg, self.test_idx, etype=etype_to_remove_1)
        # hg = dgl.remove_edges(hg, self.test_idx, etype=etype_to_remove_2)

        # self.ppi = dgl.to_homogeneous(hg, ndata = 'h').to(self.device)
        # self_loop = torch.zeros_like(self.ppi.edata['ppi'])
        # self_loop[self.ppi.edge_ids(nr_:=np.arange(self.ppi.number_of_nodes()), nr_)] = 1.0
        # self.ppi.edata['self'] = self_loop

        # self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers)
        self.train_idx = self.train_idx.to(self.device)
        self.valid_idx = self.valid_idx.to(self.device)
        self.test_idx = self.test_idx.to(self.device)
        
        # self.sampler = dgl.dataloading.NeighborSampler([-1, -1])
        # self.sampler = dgl.dataloading.NeighborSampler([5])
        self.sampler = dgl.dataloading.NeighborSampler([10, 10])
        
        self.train_loader = dgl.dataloading.DataLoader(
                    self.ppi, self.train_idx, self.sampler,
                    batch_size=batch_size, device=device, shuffle=True)
        self.valid_loader = dgl.dataloading.DataLoader(
                    self.ppi, self.valid_idx, self.sampler,
                    batch_size=batch_size, device=device, shuffle=True)
        self.test_loader = dgl.dataloading.DataLoader(
                    self.ppi, self.test_idx, self.sampler,
                    batch_size=batch_size, device=device, shuffle=True)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return 
    
    def get_norm_dgl_graph(self, g):
        in_degrees = g.in_degrees().float()
        out_degrees = g.out_degrees().float()
        in_norm = in_degrees ** -0.5
        out_norm = out_degrees ** -0.5

        in_norm_tensor = torch.diag(in_norm)
        out_norm_tensor = torch.diag(out_norm)

        adj_matrix = g.adjacency_matrix(scipy_fmt="csr")

        norm_adj_matrix = in_norm_tensor @ adj_matrix @ out_norm_tensor

        norm_g = dgl.from_scipy(norm_adj_matrix)

        return norm_g
        
    def create_full_ppi(self):
        # 添加edata['ppi']
        link = pd.read_csv(self.data_path.joinpath('link.dat'), header=None, sep='\t')
        link_test = pd.read_csv(self.data_path.joinpath('link.dat.test'), header=None, sep='\t')
        link = pd.concat([link, link_test], axis=0)
        p2g_edge_attrs = torch.tensor(link[link[2]==0][3].values, dtype=torch.float32)
        p2p_edge_attrs = torch.tensor(link[link[2]==1][3].values, dtype=torch.float32)
        g2g_edge_attrs = torch.tensor(link[link[2]==2][3].values, dtype=torch.float32)
        edge_tensor_dict = {
                ('protein', 'interacts_0', 'go_annotation'): p2g_edge_attrs,
                ('go_annotation', '_interacts_0', 'protein'): p2g_edge_attrs,
                ('protein', 'interacts_1', 'protein'): p2p_edge_attrs,
                ('protein', '_interacts_1', 'protein'): p2p_edge_attrs,
                ('go_annotation', 'interacts_2', 'go_annotation'): g2g_edge_attrs,
                ('go_annotation', '_interacts_2', 'go_annotation'): g2g_edge_attrs
            }
        # edge_tensor = torch.cat((p2g_edge_attrs, p2p_edge_attrs, g2g_edge_attrs), dim=0)
        hg = self.hg
        hg.edata['ppi'] = edge_tensor_dict
        
        sequence = torch.arange(self.go_num)
        binary_tensor = torch.zeros(self.go_num, self.feature_dim, dtype=torch.float32)
        for i in range(self.go_num):
            binary_str = bin(sequence[i])[2:]
            binary_list = [int(bit) for bit in binary_str]
            binary_tensor[i, -len(binary_list):] = torch.tensor(binary_list, dtype=torch.float32)
        feature_dict = {'protein': hg.ndata['h']['protein'], 'go_annotation': binary_tensor}
        hg.ndata['h']=feature_dict
        
        
        # 删除验证和测试数据
        etype_to_remove_1 = ('protein', 'interacts_0', 'go_annotation')
        etype_to_remove_2 = ('go_annotation', '_interacts_0', 'protein')
        hg = dgl.remove_edges(hg, self.valid_idx, etype=etype_to_remove_1)
        hg = dgl.remove_edges(hg, self.valid_idx, etype=etype_to_remove_2)
        hg = dgl.remove_edges(hg, self.test_idx, etype=etype_to_remove_1)
        hg = dgl.remove_edges(hg, self.test_idx, etype=etype_to_remove_2)
        
        # 同构化 添加自环
        g = dgl.to_homogeneous(hg, ndata=['h'], edata=['ppi'])
        nr_ = torch.tensor(np.arange(g.number_of_nodes()))
        for n in nr_:
            if not g.has_edges_between(n, n):
                g.add_edges(n, n)
                self_loop = torch.zeros_like(g.edata['ppi'])
        self_loop[g.edge_ids(nr_, nr_)] = 1.0
        g.edata['self'] = self_loop
        return g
        
        
    def create_graph(self):
        # hg, _ = dgl.load_graphs(self.hg_path)
        # hg = hg[0]
        # ppi = dgl.node_subgraph(hg, {'protein': range(hg.num_nodes('protein'))})
        # ppi = dgl.to_homogeneous(ppi, ndata = 'h')
        
        # # set self.go_num
        # self.go_num = hg.number_of_nodes('go_annotation')
        data_path = self.data_path
        ppi_path = P.Path(self.ppi_path)
        if os.path.exists(ppi_path):
            ppi, _ = dgl.load_graphs(str(ppi_path))
            return ppi[0]
        else:
            link = pd.read_csv(data_path.joinpath('link.dat'), header=None, sep='\t')
            ppi_link = link[link[2]==1]
            
            src_nodes = torch.tensor(ppi_link[0].values)
            dst_nodes = torch.tensor(ppi_link[1].values)
            ppi = dgl.graph((src_nodes, dst_nodes))
            ppi.edata['ppi'] = torch.tensor(ppi_link[3].values)
            self_loop = torch.zeros_like(ppi.edata['ppi'])
            self_loop[ppi.edge_ids(nr_:=np.arange(ppi.number_of_nodes()), nr_)] = 1.0
            ppi.edata['self'] = self_loop
            for key in ppi.edata.keys():
                ppi.edata[key] = ppi.edata[key].float()
            
            node = pd.read_csv(data_path.joinpath('node.dat'), header=None, sep='\t')
            node.columns = ['id', 'name', 'type', 'feature']
            node = node[node['type']==0]
            features = []
            for _, row in node.iterrows():
                features.append(row['feature'].split(','))
            features = torch.Tensor([list(map(float, feature)) for feature in features])
            ppi.ndata['h'] = features
            dgl.save_graphs(str(ppi_path), ppi)
        return ppi

    def get_go_num(self):
        node = pd.read_csv(self.data_path.joinpath('node.dat'), header=None, sep='\t')
        node.columns = ['id', 'name', 'type', 'feature']
        node = node[node['type']==1]
        
        return node.shape[0]
    
    def get_split(self, mode='default', validation=True):
        r"""
        
        Parameters
        ----------
        mode: 'default' or 'random'
            How to get test index.
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
            all_nodes = list(range(self.protein_num))
            train_idx = [node for node in all_nodes if node not in test_idx]
            train_idx = torch.tensor(train_idx, dtype=torch.long)  # 转为 tensor 类型
            test_idx = torch.tensor(test_idx, dtype=torch.long)  

        elif mode == 'random':
            num_nodes = self.go_num
            n_test = int(num_nodes * 0.2)
            n_train = num_nodes - n_test

            train, test = torch.utils.data.random_split(range(num_nodes), [n_train, n_test])
            train_idx = torch.tensor(train.indices)
            test_idx = torch.tensor(test.indices)
        if validation:
            random_int = torch.randperm(len(train_idx))
            valid_idx = train_idx[random_int[:len(train_idx) // 10]]
            train_idx = train_idx[random_int[len(train_idx) // 10:]]
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
            protein_num = self.protein_num
            link[1] = link[1] - protein_num
            link_test[1] = link_test[1] - protein_num

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
    
    def save_results(self, index, results, labels):
        node = pd.read_csv(self.data_path.joinpath('node.dat'), header=None, sep='\t')
        protein_names = node[2][index]
        