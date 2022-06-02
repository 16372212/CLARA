import argparse
import copy
import pickle
import random
import io
import warnings
from collections import defaultdict
import os, sys
import time

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy import sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import shuffle as skshuffle
from tqdm import tqdm

# 为了让gcc添加到源路径
warnings.filterwarnings("ignore")
root_path = os.path.abspath("./")
sys.path.append(root_path)
from gcc.Sample import Sample, Node
from gcc.datasets.data_util import create_graph_classification_dataset
from gcc.tasks import build_model

warnings.filterwarnings("ignore")

family_model_name = "./gcc/result/model_family.pickle.dat"
big_label_model_name = "./gcc/result/model_big_label.pickle.dat"


def k_label_to_q_label(label_k, q_to_k_index):
    label_q = []
    for k_index in q_to_k_index:
        label_q.append(label_k[k_index])
    return label_q


class GraphClassification(object):
    def __init__(self, dataset, model, hidden_size, num_shuffle, seed, **model_args):
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
        print(f'self labels')
        print(self.labels)
        print(f'self big labels')
        print(self.big_labels[0:20])

    def train(self):
        time_start = time.time()
        embeddings = self.model.train(None)
        time_end = time.time()
        print('get embeddings cost', time_end - time_start, 's')
        result = {'family_result': self.svc_classify(embeddings, self.labels, family_model_name, False),
                  'big_label_result': self.svc_classify(embeddings, self.big_labels, big_label_model_name, False)}
        return result

    def svc_classify(self, x, y, name, search):
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)
        accuracies = []
        recall = []
        precision = []
        f1_scores = []
        for train_index, test_index in kf.split(x, y):
            # x_train, x_test = x[train_index], x[test_index]
            # y_train, y_test = y[train_index], y[test_index]
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
            if search:
                params = {"C": [1, 10, 100, 1000, 10000, 100000]}
                classifier = GridSearchCV(
                    SVC(), params, cv=5, scoring="accuracy", verbose=0, n_jobs=-1
                )
            else:
                classifier = SVC(C=100000)
            classifier.fit(x_train, y_train)
            print(f'x shape:{x.shape}\n test_index:{test_index}\n x_test shape:{x_test.shape}')
            print(f'x shape:{x.shape}\n train_index:{train_index}\n x_train shape:{x_train.shape}')

            y_pred = classifier.predict(x_test)
            # save the classifier
            pickle.dump(classifier, open(name, "wb"))
            print('y_pred:')
            print(y_pred)
            print('y_test:')
            print(y_test)
            print(f'acc: {accuracy_score(y_test, y_pred)}')
            recall.append(recall_score(y_test, y_pred, average='weighted'))
            precision.append(precision_score(y_test, y_pred, average='weighted'))
            accuracies.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
            break
        print(f'f1-score: {f1_scores}')
        print(
            f'Micro-F1: {np.mean(accuracies)},precision:{np.mean(precision)},"recall":{np.mean(recall)},"f1-score":{np.mean(f1_scores)}')
        return {"acc": np.mean(accuracies), "precision": np.mean(precision), "recall": np.mean(recall),
                "f1-score": np.mean(f1_scores)}


# def get_test_data(model, hidden_size, num_shuffle, seed, **model_args):
#     dataset = create_graph_classification_dataset()
#     label_list = k_label_to_q_label(dataset['graph_labels'], dataset['q_to_k_index'])
#     print(f'label_list:{len(label_list)}')
#     labels = np.array(label_list)
#     print(f'labels:{labels}')
#     print("labels形状：", labels.shape)
#
#     # 加载构建好的emb模型: emb: dataset['graph_k_lists']到mydataset.npy的过程
#     model = build_model(model, hidden_size, **model_args)
#     embeddings = model.train(None)
#     print("embeddings元素总数：", embeddings.size)  # 打印数组尺寸，即数组元素总数
#     print("embeddings形状：", embeddings.shape)  # (20197, 64)
#     print("embeddings[:2]形状：", embeddings[:2].shape)
#     print("embeddings[:1]形状：", embeddings[:1].shape)
#
#     test_predict(embeddings[:1], labels[:1])
#
# def test_predict(X_test, y_test):
#     loaded_model = pickle.load(open(family_model_name, "rb"))
#     y_pred = loaded_model.predict(X_test)
#     print(f'y_pred: {y_pred}, actual value:{y_test}')


if __name__ == "__main__":
    print("-----run main func in graph classification-----")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-shuffle", type=int, default=10)
    parser.add_argument("--emb-path", type=str, default="")
    args = parser.parse_args()
    task = GraphClassification(
        args.dataset,
        args.model,
        args.hidden_size,
        args.num_shuffle,
        args.seed,
        emb_path=args.emb_path,
    )
    ret = task.train()
    print(ret)
    # write result
    f1 = open(args.emb_path.split('/')[0] + '/result.txt', 'w')
    f1.write(str(ret))
    f1.close()

    # get_test_data(args.model, args.hidden_size, args.num_shuffle, args.seed, emb_path=args.emb_path)
