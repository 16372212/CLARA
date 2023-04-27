# -*- coding: utf-8 -*-
# @Author :ZhenZiyang
import http
import os
import sys
import pickle
from typing import List
from numpy import sort
import torch
import dgl
import requests
from util.const import MID_DATA_PATH

# root_path = os.path.abspath("./")
# sys.path.append(root_path)

GRAPH_OUTPUT_PATH = f'../{MID_DATA_PATH}/graph.pkl'
NODE_OUTPUT_PATH = f'../{MID_DATA_PATH}/nodes.pkl'
SAMPLE_LIST_OUTPUT_PATH = f'../{MID_DATA_PATH}/sample_list.pkl'
API_MATRIX_OUTPUT_PATH = f'../{MID_DATA_PATH}/api_matrix.pkl'
API_INDEX_OUTPUT_PATH = f'../{MID_DATA_PATH}/api_index_map.pkl'
SAMPLE_NUM_TO_NODE_ID_PATH = f'../{MID_DATA_PATH}/sample_num_to_node_id.pkl'
DGL_OUTPUT_PATH = f'../{MID_DATA_PATH}/gcc_input/subgraphs_train_data.bin'  # 构造的dgl
GRAPH_SUB_AUG_INPUT_PATH = f'../{MID_DATA_PATH}/gcc_input/aug_graphs_n/'  # 构造的正样本的存放路径

API_LIST_LEN = 32

td = {'api': 0, 'network': 1, 'reg': 2, 'file': 3, 'process': 4}

with open(GRAPH_OUTPUT_PATH, 'rb') as fr:
    graph = pickle.load(fr)

with open(SAMPLE_LIST_OUTPUT_PATH, 'rb') as f:
    sample_list = pickle.load(f)

with open(NODE_OUTPUT_PATH, 'rb') as f:
    nodes = pickle.load(f)

with open(API_MATRIX_OUTPUT_PATH, 'rb') as f:
    api_matrix = pickle.load(f)

with open(API_INDEX_OUTPUT_PATH, 'rb') as f:
    api_index_map = pickle.load(f)

# 这个是sample num->node id的对应关系
with open(SAMPLE_NUM_TO_NODE_ID_PATH, 'rb') as f:
    sample_num_to_node_id = pickle.load(f)


def get_graph():
    """记录的是每个节点之间的连接关系"""
    # 's2552': {'network': {0, 1, 2, 6756, 6665, 10, 6667, 6670, 18617}}
    print(graph)


def get_node():
    """记录的是每个节点的属性信息，没有关系"""
    for node in nodes:
        # num: 27329, name:closesocket, type:api, sample:, pid:0, key:
        print(f"num: {node.num}, name:{node.name}, type:{node.type_}, sample:{node.sample}, pid:{node.pid}, key:{node.key}")


def get_property_maps():
    with open(f"../{MID_DATA_PATH}/property_maps.pkl", "rb") as file:
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


def get_label_familys():
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

    return label_familys


def client_backend_create_category():
    """
    构造http请求
    curl --location --request POST 'http://127.0.0.1:8081/categories' \
    --header 'Content-Type: application/json' \
    --header 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJVc2VySWQiOjE4LCJleHAiOjE2NjU3NTk0MzMsImlhdCI6MTY2NTE1NDYzMywiaXNzIjoiamtkZXYuY24iLCJzdWIiOiJ1c2VyIHRva2VuIn0.qtYlsrXU4WBJHoSeo8vKUN1oQs9E4j85VU52rR1gjiM' \
    --data-raw '{
        "label_name": "-1",
        "family_name": "backdore2"
    }'
    """
    url = "http://127.0.0.1:8081/categories"
    headers = {'Authorization': 'Bearer 123',
               'Content-Type': 'application/json'}
    label_familys = get_label_familys()
    for label in label_familys:
        for family in label_familys[label]:
            payload = {"label_name": label, "family_name": family}
            r = requests.post(url, headers=headers, json=payload)
            print(f'family: {label}, label: {family}')
            print(
                f'status code: {r.status_code}, \n {str(r.content, encoding="utf-8")}')


def analyze_sample():
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

    print(sorted(actual_labels.items(), key=lambda x: x[1]))
    for label in actual_labels:
        print("{ value: " + str(actual_labels[label]) + ", name: "+label + " },")

    print(f'actual collected families this time: {len(actual_familys)}')
    print(actual_familys)
    print(actual_labels)

    # data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    total_family_num = 0
    for l in label_familys:
        print(f'family: {l}, {len(label_familys[l])}, label: {label_familys[l]}')
#        print(f'{l}: {len(label_familys[l])}')
        total_family_num += len(label_familys[l])
    print(f'family total {total_family_num}')


if __name__ == "__main__":
    analyze_sample()
