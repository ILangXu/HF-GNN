import traceback

import networkx as nx
import torch
from rdkit import Chem #RDKit 中的分子默认采用隐式H原子形式
import dgl
from scipy.spatial import distance_matrix
from dgllife.utils import BaseAtomFeaturizer, atom_type_one_hot, atom_degree_one_hot, atom_total_num_H_one_hot, \
    atom_is_aromatic, ConcatFeaturizer, bond_type_one_hot, atom_hybridization_one_hot, \
    one_hot_encoding, atom_formal_charge, atom_num_radical_electrons, bond_is_conjugated, \
    bond_is_in_ring, bond_stereo_one_hot, BaseBondFeaturizer, mol_to_bigraph
from functools import partial
from itertools import repeat
from torchani import SpeciesConverter, AEVComputer
import multiprocessing as mp
from prody import *
from pylab import *
import pandas as pd
import warnings
import os
import pickle
import shutil
from Hyper_utils.func_group_helpers import get_graph,ATOM_FEATURIZER,BOND_FEATURIZER,NODE_ATTRS,EDGE_ATTRS
warnings.filterwarnings('ignore')
converter = SpeciesConverter(['C', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I'])
add_self_loop=False
node_featurizer=ATOM_FEATURIZER
edge_featurizer=BOND_FEATURIZER
canonical_atom_order=True
explicit_hydrogens=False
mean_fg_init: bool = False
use_cycle: bool = False
fully_connected_fg: bool = False
num_virtual_nodes=0

def chirality(atom):  # the chirality information defined in the AttentiveFP
    try:
        return one_hot_encoding(atom.GetProp('_CIPCode'), ['R', 'S']) + \
               [atom.HasProp('_ChiralityPossible')]
    except:
        return [False, False] + [atom.HasProp('_ChiralityPossible')]


class MyAtomFeaturizer(BaseAtomFeaturizer):
    """
        原子特征化器的抽象类。
       循环一个分子中的所有原子，并使用featurer_funcs对其进行特征化。
       我们假设生成的DGLGraph不包含任何虚拟节点，并且图中的节点i正好对应于分子中的原子i。
       """
    def __init__(self, atom_data_filed='h'):
        super(MyAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_filed: ConcatFeaturizer([partial(atom_type_one_hot, #functools.partial 这个高阶函数用于部分应用一个函数。 部分应用是指， 基于一个函数创建一个新的可调用对象， 把原函数的某些参数固定。 使用这个函数可以把接受一个或多个参数的函数改编成需要回调的API， 这样参数更少。
                                                                         allowable_set=['C', 'N', 'O', 'S', 'F', 'P',
                                                                                        'Cl', 'Br', 'I', 'B', 'Si',
                                                                                        'Fe', 'Zn', 'Cu', 'Mn', 'Mo'],
                                                                         encode_unknown=True),#返回最多有一个值为True的布尔值列表
                                                                 partial(atom_degree_one_hot, #原子的度被定义为其直接键合的相邻原子的数量。
                                                                         allowable_set=list(range(6))),
                                                                 atom_formal_charge, atom_num_radical_electrons,#获取原子电荷，得到一个原子的自由基电子数。
                                                                 partial(atom_hybridization_one_hot,
                                                                         encode_unknown=True),
                                                                 atom_is_aromatic,
                                                                 # A placeholder for aromatic information,
                                                                 atom_total_num_H_one_hot, chirality])})#atom_total_num_H_one_hot该原子连接的氢的总数


class MyBondFeaturizer(BaseBondFeaturizer):
    def __init__(self, bond_data_filed='e'):
        super(MyBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_filed: ConcatFeaturizer([bond_type_one_hot, bond_is_conjugated, bond_is_in_ring,
                                                                 partial(bond_stereo_one_hot, allowable_set=[
                                                                     Chem.rdchem.BondStereo.STEREONONE,
                                                                     Chem.rdchem.BondStereo.STEREOANY,
                                                                     Chem.rdchem.BondStereo.STEREOZ,
                                                                     Chem.rdchem.BondStereo.STEREOE],
                                                                         encode_unknown=True)])})


def D3_info(a, b, c):
    # 空间夹角
    ab = b - a  # 向量ab
    ac = c - a  # 向量ac
    cosine_angle = np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac))
    cosine_angle = cosine_angle if cosine_angle >= -1.0 else -1.0
    angle = np.arccos(cosine_angle)
    # 三角形面积
    ab_ = np.sqrt(np.sum(ab ** 2))
    ac_ = np.sqrt(np.sum(ac ** 2))  # 欧式距离
    area = 0.5 * ab_ * ac_ * np.sin(angle)
    return np.degrees(angle), area, ac_


# claculate the 3D info for each directed edge
def D3_info_cal(nodes_ls, g):
    if len(nodes_ls) > 2:
        Angles = []
        Areas = []
        Distances = []
        for node_id in nodes_ls[2:]:
            angle, area, distance = D3_info(g.ndata['pos'][nodes_ls[0]].numpy(), g.ndata['pos'][nodes_ls[1]].numpy(),
                                            g.ndata['pos'][node_id].numpy())
            Angles.append(angle)
            Areas.append(area)
            Distances.append(distance)
        return [np.max(Angles) * 0.01, np.sum(Angles) * 0.01, np.mean(Angles) * 0.01, np.max(Areas), np.sum(Areas),
                np.mean(Areas),
                np.max(Distances) * 0.1, np.sum(Distances) * 0.1, np.mean(Distances) * 0.1]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]


AtomFeaturizer = MyAtomFeaturizer()
BondFeaturizer = MyBondFeaturizer()


def graphs_from_mol_ign(dir, key, label, graph_dic_path, dis_threshold=8.0, path_marker='/',
                        EtaR=4.00, ShfR=3.17, Zeta=8.00, ShtZ=3.14):
    """
       pool.starmap(partial(graphs_from_mol_ign, graph_dic_path=self.graph_dic_path,
                                 dis_threshold=self.dis_threshold, path_marker=self.path_marker,
                                 EtaR=self.EtaR, ShfR=self.ShfR, Zeta=self.Zeta, ShtZ=self.ShtZ),
                         zip(self.origin_data_dirs, self.origin_keys, self.origin_labels))
    This function is used for generating graph objects using multi-process for ign model training
    :param dir: the absolute path for the complex (ligand + pocket)复合物的绝对路径（配体+口袋）
    :param key: the key for the complex 复合体名称
    :param label: the label for the complex 标签
    :param dis_threshold: the distance threshold to determine the atom-pair interactions 确定原子对相互作用的距离阈值
    :param graph_dic_path: the absolute path for storing the generated graph 用于存储生成的图形的绝对路径
    :param path_marker: '\\' for window and '/' for linux
    :param EtaR: acsf parameter
    :param ShfR: acsf parameter
    :param Zeta: acsf parameter
    :param ShtZ: acsf parameter
    :return:
    """
    add_self_loop = False
    try:
        with open(dir, 'rb') as f:
            mol1, mol2 = pickle.load(f) #读取复合体，mol1是配体，mol2是蛋白质
        # the distance threshold to determine the interaction between ligand atoms and protein atoms
        dis_threshold = dis_threshold

        # construct graphs1
        g = dgl.DGLGraph()#是图的基类，图存储顶点，边和他们的特征
        # add nodes
        num_atoms_m1 = mol1.GetNumAtoms()  # number of ligand atoms
        num_atoms_m2 = mol2.GetNumAtoms()  # number of pocket atoms
        num_atoms = num_atoms_m1 + num_atoms_m2
        g.add_nodes(num_atoms)#设置总的节点数
        g_l = mol_to_bigraph(mol1, add_self_loop, node_featurizer, edge_featurizer,
                             canonical_atom_order, explicit_hydrogens, num_virtual_nodes)
        g_p = mol_to_bigraph(mol2, add_self_loop, node_featurizer, edge_featurizer,
                             canonical_atom_order, explicit_hydrogens, num_virtual_nodes)
        nx_multi_l = g_l.to_networkx(node_attrs=NODE_ATTRS, edge_attrs=EDGE_ATTRS).to_undirected()
        nx_multi_p = g_p.to_networkx(node_attrs=NODE_ATTRS, edge_attrs=EDGE_ATTRS).to_undirected()
        nx_g_l = nx.Graph(nx_multi_l)
        nx_g_p = nx.Graph(nx_multi_p)
        Hyper_index_l = get_graph([nx_g_l], False)[0]
        Hyper_index_p = get_graph([nx_g_p], False)[0]
        atg_l, gta_l = Hyper_index_l.unbind(dim=0)
        atg_p, gta_p = Hyper_index_p.unbind(dim=0)
        atg_p = atg_p + max(atg_l.tolist())
        gta_p = gta_p + max(gta_l.tolist())
        atg = torch.cat([atg_l,atg_p],dim=-1)
        gta = torch.cat([gta_l,gta_p],dim=-1)
        a2f_edges = (atg.long(), gta.long())  # 形成一个元组，里面有节点和超边tensor

        if add_self_loop:#是否加入自循环
            nodes = g.nodes()
            g.add_edges(nodes, nodes)

        # add edges, ligand molecule
        num_bonds1 = mol1.GetNumBonds()#获取配体分子的共价键
        src1 = []
        dst1 = []
        for i in range(num_bonds1):
            bond1 = mol1.GetBondWithIdx(i)#按照index获取共价键
            u = bond1.GetBeginAtomIdx() #得到共价键的起始原子
            v = bond1.GetEndAtomIdx()#结束原子
            src1.append(u)
            dst1.append(v)
        src_ls1 = np.concatenate([src1, dst1])#按维度连接，默认是0，由于src1和dst1是（n,）,所以0维就还是相当于列相加，就是把两个list合起来了
        dst_ls1 = np.concatenate([dst1, src1])
        g.add_edges(src_ls1, dst_ls1)#构建邻接矩阵

        # add edges, pocket
        num_bonds2 = mol2.GetNumBonds()
        src2 = []
        dst2 = []
        for i in range(num_bonds2):
            bond2 = mol2.GetBondWithIdx(i)
            u = bond2.GetBeginAtomIdx()
            v = bond2.GetEndAtomIdx()
            src2.append(u + num_atoms_m1)
            dst2.append(v + num_atoms_m1)
        src_ls2 = np.concatenate([src2, dst2])
        dst_ls2 = np.concatenate([dst2, src2])
        g.add_edges(src_ls2, dst_ls2)

        # add interaction edges, only consider the euclidean distance within dis_threshold
        g3 = dgl.DGLGraph()
        g3.add_nodes(num_atoms)#加入全部的蛋白质和配体原子
        dis_matrix = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())#得到距离矩阵
        node_idx = np.where(dis_matrix < dis_threshold)#获得两个list，1是行下标，2是列下标
        src_ls3 = np.concatenate([node_idx[0]])#构建邻接矩阵
        dst_ls3 = np.concatenate([node_idx[1] + num_atoms_m1])
        g3.add_edges(src_ls3, dst_ls3)

        # assign atom features
        # 'h', features of atoms
        g.ndata['h'] = torch.zeros(num_atoms, AtomFeaturizer.feat_size('h'), dtype=torch.float)  # init 'h'
        g.ndata['h'][:num_atoms_m1] = AtomFeaturizer(mol1)['h']
        g.ndata['h'][-num_atoms_m2:] = AtomFeaturizer(mol2)['h']

        # TorchANI package
        # acsf 计算时将蛋白-配体复合物放在一起考虑
        AtomicNums = [] #获取原子序数列表
        for i in range(num_atoms_m1):
            AtomicNums.append(mol1.GetAtomWithIdx(i).GetAtomicNum())
        for j in range(num_atoms_m2):
            AtomicNums.append(mol2.GetAtomWithIdx(j).GetAtomicNum())
        Corrds = np.concatenate([mol1.GetConformer().GetPositions(), mol2.GetConformer().GetPositions()], axis=0)#两个分子的原子3d坐标集连接
        AtomicNums = torch.tensor(AtomicNums, dtype=torch.long)
        Corrds = torch.tensor(Corrds, dtype=torch.float64)
        AtomicNums = torch.unsqueeze(AtomicNums, dim=0) #返回一个新的张量，对输入的既定位置插入维度 1，本来是(n,),现在是(1,n)
        Corrds = torch.unsqueeze(Corrds, dim=0)
        res = converter((AtomicNums, Corrds)) #???
        pbsf_computer = AEVComputer(Rcr=12.0, Rca=12.0, EtaR=torch.tensor([EtaR]), ShfR=torch.tensor([ShfR]),
                                    EtaA=torch.tensor([3.5]), Zeta=torch.tensor([Zeta]),
                                    ShfA=torch.tensor([0]), ShfZ=torch.tensor([ShtZ]), num_species=9)
        outputs = pbsf_computer((res.species, res.coordinates))
        if torch.any(torch.isnan(outputs.aevs[0].float())):
            print(key)
        g.ndata['h'] = torch.cat([g.ndata['h'], outputs.aevs[0].float()], dim=-1)

        # assign edge features
        # 'd', distance between ligand atoms
        dis_matrix_L = distance_matrix(mol1.GetConformers()[0].GetPositions(), mol1.GetConformers()[0].GetPositions())
        m1_d = torch.tensor(dis_matrix_L[src_ls1, dst_ls1], dtype=torch.float).view(-1, 1)

        # 'd', distance between pocket atoms
        dis_matrix_P = distance_matrix(mol2.GetConformers()[0].GetPositions(), mol2.GetConformers()[0].GetPositions())
        m2_d = torch.tensor(dis_matrix_P[src_ls2 - num_atoms, dst_ls2 - num_atoms_m1], dtype=torch.float).view(-1, 1)

        # 'd', distance between ligand atoms and pocket atoms
        inter_dis = np.concatenate([dis_matrix[node_idx[0], node_idx[1]]])
        g3_d = torch.tensor(inter_dis, dtype=torch.float).view(-1, 1)

        # efeats1
        g.edata['e'] = torch.zeros(g.number_of_edges(), BondFeaturizer.feat_size('e'), dtype=torch.float)  # init 'e'
        efeats1 = BondFeaturizer(mol1)['e']  # 重复的边存在！
        g.edata['e'][g.edge_ids(src_ls1, dst_ls1)] = torch.cat([efeats1[::2], efeats1[::2]])

        # efeats2
        efeats2 = BondFeaturizer(mol2)['e']  # 重复的边存在！
        g.edata['e'][g.edge_ids(src_ls2, dst_ls2)] = torch.cat([efeats2[::2], efeats2[::2]])

        # 'e'
        g1_d = torch.cat([m1_d, m2_d])
        g.edata['e'] = torch.cat([g.edata['e'], g1_d * 0.1], dim=-1)
        g3.edata['e'] = g3_d * 0.1

        # init 'pos'
        g.ndata['pos'] = torch.zeros([g.number_of_nodes(), 3], dtype=torch.float)
        g.ndata['pos'][:num_atoms_m1] = torch.tensor(mol1.GetConformers()[0].GetPositions(), dtype=torch.float)
        g.ndata['pos'][-num_atoms_m2:] = torch.tensor(mol2.GetConformers()[0].GetPositions(), dtype=torch.float)
        # calculate the 3D info for g
        src_nodes, dst_nodes = g.find_edges(range(g.number_of_edges()))
        src_nodes, dst_nodes = src_nodes.tolist(), dst_nodes.tolist()
        neighbors_ls = []
        for i, src_node in enumerate(src_nodes):
            tmp = [src_node, dst_nodes[i]]  # the source node id and destination id of an edge
            neighbors = g.predecessors(src_node).tolist()#根据节点索引获取该节点所有的前驱，由于是无向图，所以是所有的邻居
            neighbors.remove(dst_nodes[i])#把dst_nodes[i]去掉
            tmp.extend(neighbors)
            neighbors_ls.append(tmp)
        D3_info_ls = list(map(partial(D3_info_cal, g=g), neighbors_ls))
        D3_info_th = torch.tensor(D3_info_ls, dtype=torch.float)
        g.edata['e'] = torch.cat([g.edata['e'], D3_info_th], dim=-1)
        g.ndata.pop('pos') #表示移除DGL图中节点的‘pos’特征
        # detect the nan values in the D3_info_th
        if torch.any(torch.isnan(D3_info_th)):
            status = False
        else:
            status = True
        g_index_u,g_index_v = g.edges()
        hyper_g = dgl.heterograph({  # 异构图，包含了普通图和超图
            ('atom', 'interacts', 'atom'): (g_index_u.long(), g_index_v.long()),  # 节点与节点之间的交互，两个tensor，里面的值都是节点的索引
            ('atom', 'to', 'func_group'): a2f_edges,  # 节点对官能团的索引，两个tensor，tensor中的每一个值都是索引值
            # 官能团索引的组合，每一个都两两组合，两个tensor
        })
        hyper_g.nodes['atom'].data['h'] = g.ndata['h']
        hyper_g.edges[('atom', 'interacts', 'atom')].data['e'] = g.edata['e']
        num_fg = hyper_g.number_of_nodes('func_group')  # 官能团数量
        node_feat_dim = g.ndata['h'].shape[1]  # 获取特征维度
        hyper_g.nodes['func_group'].data['feat'] = torch.zeros(num_fg, node_feat_dim)
    except Exception as e:
        print(dir)
        print(traceback.format_exc())
        hyper_g = None
        g3 = None
        status = False
    if status:
        with open(graph_dic_path + path_marker + key, 'wb') as f:
            pickle.dump({'hyper_g': hyper_g, 'g3': g3, 'key': key, 'label': label}, f)


class GraphDatasetHFGNN(object):
    """
    This class is used for generating graph objects featurizerized by PBSF(Behler−Parrinello symmetric functions) using multi process
    prepared for IGN model training
    该类用于使用为IGN模型训练准备的多过程生成PBSF(Behler−Parrinello对称函数)特征化的图对象
    """

    def __init__(self, keys=None, labels=None, data_dirs=None, graph_ls_file=None, graph_dic_path=None, num_process=6,
                 dis_threshold=8.00, path_marker='/', EtaR=4.00, ShfR=3.17, Zeta=8.00, ShtZ=3.14):
        """
        :param keys: the keys for the complexes, list
        :param labels: the corresponding labels for the complexes, list
        :param data_dirs: the corresponding data_dirs for the complexes, list
        :param graph_ls_file: the cache path for the final .bin file containing graphs, graphs3, labels, keys 包含图、graphs3、标签、键的最终.bin文件的缓存路径
        :param graph_dic_path: the cache path for the separate graphs objects (dic) for each complex, do not share the same with graph_ls_file每个复合物的独立图形对象(dic)的缓存路径与graph_ls_file不相同
        :param num_process: the number of process used to generate the graph objects
        :param dis_threshold: the distance threshold for determining the atom-pair interactions
        :param path_marker: '\\' for windows and '/' for linux
        :param EtaR: acsf parameter
        :param ShfR: acsf parameter
        :param Zeta: acsf parameter
        :param ShtZ: acsf parameter
        """
        self.origin_keys = keys #复合体名称
        self.origin_labels = labels #复合体标签
        self.origin_data_dirs = data_dirs ##input_data/user1/complexes/复合体名称
        self.graph_ls_file = graph_ls_file
        self.graph_dic_path = graph_dic_path
        self.num_process = num_process
        self.dis_threshold = dis_threshold
        self.path_marker = path_marker
        self.EtaR = EtaR
        self.ShfR = ShfR
        self.Zeta = Zeta
        self.ShtZ = ShtZ
        self._pre_process()

    def _pre_process(self):
        if os.path.exists(self.graph_ls_file):
            print('Loading previously saved dgl graphs and corresponding data...')
            with open(self.graph_ls_file,'rb') as f:
                data = pickle.load(f)
            self.hyper_gs = data['hyper_g']
            self.graphs3 = data['g3']
            self.keys = data['keys']
            self.labels = data['labels']
        else:
            # mk dic path
            if os.path.exists(self.graph_dic_path)!=True:
                os.mkdir(self.graph_dic_path)
            #linux
            # cmdline = 'mkdir -p %s' % self.graph_dic_path
            # os.system(cmdline)

            print('Generate complex graph...')

            pool = mp.Pool(self.num_process)
            pool.starmap(partial(graphs_from_mol_ign, graph_dic_path=self.graph_dic_path,
                                 dis_threshold=self.dis_threshold, path_marker=self.path_marker,
                                 EtaR=self.EtaR, ShfR=self.ShfR, Zeta=self.Zeta, ShtZ=self.ShtZ),
                         zip(self.origin_data_dirs, self.origin_keys, self.origin_labels))
            pool.close()
            pool.join()
            # collect the generated graph for each complex
            self.hyper_gs = []
            self.graphs3 = []
            self.labels = []
            self.keys = os.listdir(self.graph_dic_path)
            for key in self.keys:
                with open(self.graph_dic_path + self.path_marker + key, 'rb') as f:
                    graph_dic = pickle.load(f)
                    self.hyper_gs.append(graph_dic['hyper_g'])
                    self.graphs3.append(graph_dic['g3'])
                    self.labels.append(graph_dic['label'])
            with open(self.graph_ls_file, 'wb') as f:
                pickle.dump({'hyper_g': self.hyper_gs, 'g3': self.graphs3, 'keys': self.keys, 'labels': self.labels}, f)

            # delete the temporary files
            # cmdline = 'rm -rf %s' % self.graph_dic_path  # graph_dic_path
            # os.system(cmdline)
            shutil.rmtree(self.graph_dic_path)

    def __getitem__(self, indx):
        return self.hyper_gs[indx], self.graphs3[indx], torch.tensor(self.labels[indx], dtype=torch.float), self.keys[indx]

    def __len__(self):
        return len(self.labels)


def collate_fn_hfgnn(data_batch):
    hyper_graphs, graphs3, labels, keys= map(list, zip(*data_batch))
    hyper_gs = dgl.batch(hyper_graphs)
    g3 = dgl.batch(graphs3)
    labels = torch.unsqueeze(torch.stack(labels, dim=0), dim=-1)

    return hyper_gs, g3, labels, keys