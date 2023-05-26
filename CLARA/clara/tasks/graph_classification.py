import argparse
import pickle
import warnings
import os, sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier


# 为了让gcc添加到源路径
warnings.filterwarnings("ignore")
root_path = os.path.abspath("./")
sys.path.append(root_path)
from clara.datasets.data_util import create_graph_classification_dataset
from clara.tasks import build_model

warnings.filterwarnings("ignore")

family_model_name = "./clara/result/model_family.pickle.dat"
big_label_model_name = "./clara/result/model_big_label.pickle.dat"


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
        # print(f'self labels')
        # print(self.labels)
        # print(f'self big labels')
        # print(self.big_labels[0:20])

    def get_emb(self):
        time_start = time.time()
        embeddings = self.model.train(None)
        return embeddings

    def train(self):
        time_start = time.time()
        embeddings = self.model.train(None)
        time_end = time.time()
        print('get embeddings cost', time_end - time_start, 's')
        result = {'family_result': self.svc_classify(embeddings, self.labels, family_model_name, False),
                  }
        return result


    def zsl_classify_novel(self, x, y, name, meta_embeddings_all):
        """测试集中的类没有在训练集中出现"""
        unique_labels = np.unique(y)
        accuracies = []
        recall = []
        precision = []
        f1_scores = []
        np.random.shuffle(unique_labels)
        num_train_labels = int(len(unique_labels) * 0.8)
        train_labels = unique_labels[:num_train_labels]
        test_labels = unique_labels[num_train_labels:]

        train_indices = [i for i, label in enumerate(y) if label in train_labels]
        test_indices = [i for i, label in enumerate(y) if label in test_labels]

        x_train, y_train = x[train_indices], y[train_indices]
        x_test, y_test = x[test_indices], y[test_indices]

        n_clusters = len(np.unique(y_train))
        classifier = KMeans(n_clusters=n_clusters, random_state=0).fit(x_train)
        y_pred = classifier.predict(x_test)

        # Note: accuracy_score, recall_score, precision_score, f1_score
        # might not be meaningful for unsupervised methods like KMeans.
        # I suggest to use adjusted_rand_score or other metrics designed for unsupervised learning
        ari = adjusted_rand_score(y_test, y_pred)
        accuracies.append(ari)

        print(f'Adjusted Rand index: {np.mean(accuracies)}')
        return {"ARI": np.mean(accuracies)}


    def svc_classify(self, x, y, name, search):
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)
        accuracies = []
        recall = []
        precision = []
        f1_scores = []
        for train_index, test_index in kf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
            if search:
                params = {"C": [1, 10, 100, 1000, 10000, 100000]}
                classifier = GridSearchCV(
                    SVC(), params, cv=5, scoring="accuracy", verbose=0, n_jobs=-1
                )
            else:
                classifier = SVC(C=100000)
            classifier.fit(x_train, y_train)

            y_pred = classifier.predict(x_test)
            # save the classifier
            pickle.dump(classifier, open(name, "wb"))

            recall.append(recall_score(y_test, y_pred, average='weighted'))
            precision.append(precision_score(y_test, y_pred, average='weighted'))
            accuracies.append(accuracy_score(y_test, y_pred))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
            # break
        # print(f'f1-score: {f1_scores}')
        print(
            f'acc: {np.mean(accuracies)},precision:{np.mean(precision)},"recall":{np.mean(recall)},"f1-score":{np.mean(f1_scores)}')

        # with open(name, 'wb') as fr:
        #     pickle.dump(conf_mat, fr)

        return {"acc": np.mean(accuracies), "precision": np.mean(precision), "recall": np.mean(recall),
                "f1-score": np.mean(f1_scores)}


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
    print("result will write to " + args.emb_path.split('/')[1])
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
    f1 = open(args.emb_path.split('/')[1] + '/result.txt', 'w')
    f1.write(str(ret))
    f1.close()

    # with open('./label_matrix', 'rb') as fr:
    #     conf_mat = pickle.load(fr)
    # print(conf_mat)
    # conf_mat = np.array(conf_mat, dtype=int)
    #
