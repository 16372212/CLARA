# -*- coding: utf-8 -*-
# @Author :ZhenZiyang
import os
import sys
import pickle
from typing import List

from numpy import sort

from Sample import Node, Sample
import torch
import dgl

# root_path = os.path.abspath("./")
# sys.path.append(root_path)

GRAPH_OUTPUT_PATH = '../mid_data/graph.pkl'
NODE_OUTPUT_PATH = '../mid_data/nodes.pkl'
SAMPLE_LIST_OUTPUT_PATH = '../mid_data/sample_list.pkl'
API_MATRIX_OUTPUT_PATH = '../mid_data/api_matrix.pkl'
API_INDEX_OUTPUT_PATH = '../mid_data/api_index_map.pkl'
SAMPLE_NUM_TO_NODE_ID_PATH = '../mid_data/sample_num_to_node_id.pkl'
DGL_OUTPUT_PATH = '../mid_data/gcc_input/subgraphs_train_data.bin'  # 构造的dgl
GRAPH_SUB_AUG_INPUT_PATH = '../mid_data/gcc_input/aug_graphs_15/'  # 构造的正样本的存放路径

API_LIST_LEN = 32

td = {'api': 0, 'network': 1, 'reg': 2, 'file': 3, 'process': 4}

with open(GRAPH_OUTPUT_PATH, 'rb') as fr:
    graph = pickle.load(fr)

with open(SAMPLE_LIST_OUTPUT_PATH, 'rb') as f:
    sample_list = pickle.load(f)

with open(NODE_OUTPUT_PATH, 'rb') as f:
    Nodes = pickle.load(f)

with open(API_MATRIX_OUTPUT_PATH, 'rb') as f:
    api_matrix = pickle.load(f)

with open(API_INDEX_OUTPUT_PATH, 'rb') as f:
    api_index_map = pickle.load(f)

# 这个是sample num->node id的对应关系
with open(SAMPLE_NUM_TO_NODE_ID_PATH, 'rb') as f:
    sample_num_to_node_id = pickle.load(f)


def get_graph():
    print(Nodes)


def get_property_maps():
    with open("../mid_data/property_maps.pkl", "rb") as file:
        property_maps = pickle.load(file)

    print(api_matrix)
    return property_maps


def get_dgl_property():
    """
    根据各类属性得到dgl中节点
        api: input: api_name, output api_array = api_matrix[api_index_map[api_name]]

    :return:
    """

    # 构建dgl_graph
    dgl_graph = dgl.DGLGraph((torch.tensor(left_nodes), torch.tensor(right_nodes)))
    node_type, api_pro = gen_ndata_property()  # 根据Nodes的顺序得到的，未筛选
    # 构造dgl节点的三种属性
    dgl_graph.ndata['node_type'] = torch.tensor(node_type)
    dgl_graph.ndata['api_pro'] = torch.tensor(np.array(api_pro))
    # dgl_graph.ndata['node_id'] = torch.tensor([*range(0, len(node_type), 1)])


def analyze_sample():
    SAMPLE_LIST_OUTPUT_PATH = '../mid_data/sample_list.pkl'

    with open(SAMPLE_LIST_OUTPUT_PATH, 'rb') as f:
        sample_list = pickle.load(f)

    print(f'actual sample lists: {len(sample_list)}')
    actual_familys = {}
    actual_labels = {}
    label_familys = {}
    for samples in sample_list:
        if samples.label not in actual_labels:
            actual_labels[samples.label] = 1
            label_familys[samples.label] = set()
        else:
            actual_labels[samples.label] += 1

        if samples.family not in actual_familys:
            actual_familys[samples.family] = 1
        else:
            actual_familys[samples.family] += 1

        label_familys[samples.label].add(samples.family)

    print(sorted(actual_labels.items(), key=lambda kv: (kv[1], kv[0])))

    print(f'actual collected families this time: {len(actual_familys)}')
    print(actual_familys)
    print(actual_labels)

    for l in label_familys:
        print(f'{l}: {len(label_familys[l])}')


if __name__ == "__main__":
    analyze_sample()
