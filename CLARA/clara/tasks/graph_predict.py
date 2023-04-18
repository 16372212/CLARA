import argparse
import pickle
import warnings
import os, sys
import numpy as np
import torch

# 为了让gcc添加到源路径
warnings.filterwarnings("ignore")
root_path = os.path.abspath("./")
sys.path.append(root_path)
from clara.datasets.data_util import create_graph_classification_dataset
from clara.tasks import build_model
from clara.models import GraphEncoder

GRAPH_OUTPUT_PATH = 'clara/result/'

family_model_name = "./clara/result/model_family.pickle.dat"
big_label_model_name = "./clara/result/model_big_label.pickle.dat"

emb_model_name = "./Pretrain_mydataset_layer_5_bsz_32_nce_k_16384_momentum_0.99_r0.15/current.pth"


def k_label_to_q_label(label_k, q_to_k_index):
    label_q = []
    for k_index in q_to_k_index:
        label_q.append(label_k[k_index])
    return label_q


def get_emb(file_info):
    checkpoint = torch.load(emb_model_name, map_location="cpu")
    # create model and optimizer
    model = GraphEncoder(
        positional_embedding_size=args.positional_embedding_size,
        max_node_freq=args.max_node_freq,
        max_edge_freq=args.max_edge_freq,
        max_degree=args.max_degree,
        freq_embedding_size=args.freq_embedding_size,
        degree_embedding_size=args.degree_embedding_size,
        output_dim=args.hidden_size,
        node_hidden_dim=args.hidden_size,
        edge_hidden_dim=args.hidden_size,
        num_layers=args.num_layer,
        num_step_set2set=args.set2set_iter,
        num_layer_set2set=args.set2set_lstm_layer,
        gnn_model=args.model,
        norm=args.norm,
        degree_input=True,
    )
    model = model.to(torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])

    model.eval()
    dgl_graph = parse_file2dgl(file_info)
    with torch.no_grad():
        outputs = model(dgl_graph)

    return outputs


def parse_file2dgl(file_info):
    # TODO 1: 调用接口，file -> 行为信息 2。dgl = []

    # TODO 2: 对应api的id，属性id，找到Nodes，api_matrix中名字对应的index，

    # TODO 3: 构造dgl
    dgl_graph = dgl.DGLGraph((torch.tensor(left_nodes), torch.tensor(right_nodes)))
    node_type = [], api_pro = []  # gen_ndata_property(Nodes)
    # 构造dgl节点的三种属性
    dgl_graph.ndata['node_type'] = torch.tensor(node_type)
    dgl_graph.ndata['api_pro'] = torch.tensor(np.array(api_pro))
    dgl_graph.ndata['node_id'] = torch.tensor([*range(0, len(node_type), 1)])
    return dgl_graph


def client():


def svc_predict(x, model_path):
    loaded_model = pickle.load(open(model_path, "rb"))
    y_pred = loaded_model.predict(x)
    print(f'y_pred: {y_pred}')
    return y_pred


class GraphClassification(object):
    def __init__(self, dataset, model, hidden_size, num_shuffle, seed, input_file, **model_args):
        assert model == "from_numpy_graph"
        dataset = create_graph_classification_dataset()
        self.num_nodes = len(dataset['graph_labels'])
        self.num_classes = dataset['num_labels']
        self.label_matrix = np.zeros((self.num_nodes, self.num_classes), dtype=int)
        self.labels = np.array(k_label_to_q_label(dataset['graph_labels'], dataset['q_to_k_index']))
        # self.labels = np.array(dataset.graph_labels)
        self.big_labels = np.array(k_label_to_q_label(dataset['graph_big_labels'], dataset['q_to_k_index']))
        self.model = build_model(model, hidden_size, **model_args)
        self.hidden_size = hidden_size
        self.num_shuffle = num_shuffle
        self.seed = seed
        self.test_graphs = []
        self.file = input_file
        print(f'self labels')
        print(self.labels)
        print(f'self big labels')
        print(self.big_labels[0:20])

    def predict(self):
        embeddings = get_emb(self.file)
        result = {'family_result': svc_predict(embeddings, family_model_name),
                  'big_label_result': svc_predict(embeddings, big_label_model_name)}
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shuffle", type=int, default=10)
    parser.add_argument("--emb-path", type=str, default="")
    args = parser.parse_args()
    files = []  # TODO
    task = GraphClassification(
        args.dataset,
        args.model,
        args.hidden_size,
        args.num_shuffle,
        args.seed,
        files,
        emb_path=args.emb_path,
    )
    ret = task.predict()
    print(ret)
    # write result
