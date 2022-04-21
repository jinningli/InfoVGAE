import numpy as np
import pickle
import json
import time
from sklearn import metrics

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score


class Evaluator:
    def __init__(self, user_plot_config=None, asser_plot_config=None):
        self.initialized = False
        self.output_dir = None
        self.embedding = None
        self.labels = None
        self.user_label = None
        self.asser_label = None
        self.user_clustering_pred = None
        self.asser_clustering_pred = None
        self.namelist = None
        self.asserlist = None
        self.num_user = -1
        self.num_asser = -1
        self.user_plot_config = [
            [1, "#178bff", "User Con"],
            # [0, "#3b3b3b", "User Neu"],
            [2, "#ff5c1c", "User Pro"]
        ] if user_plot_config is None else user_plot_config
        self.asser_plot_config = [
            [1, "#30a5ff", "Assertion Con"],
            # [0, "#4d4d4d", "Assertion Neu"],
            [2, "#fc8128", "Assertion Pro"]
        ] if asser_plot_config is None else asser_plot_config

    def initialize(self):
        if self.labels is not None:
            self.user_label = self.labels["user_label"]
            self.asser_label = self.labels["assertion_label"]
        self.num_user = self.user_label.shape[0]
        self.num_asser = self.asser_label.shape[0]

        for item in self.asser_plot_config:
            item[0] += 1000
        self.asser_label += 1000

        self.initialized = True

    def init_from_dir(self, dir):
        with open(dir + "/embedding.bin", 'rb') as fin:
            self.embedding = pickle.load(fin)
        with open(dir + "/label.bin", 'rb') as fin:
            self.labels = pickle.load(fin)
        with open(dir + "/namelist.json", 'r') as fin:
            self.namelist = json.load(fin)
        with open(dir + "/asserlist.json", 'r') as fin:
            self.asserlist = json.load(fin)
        self.output_dir = dir
        self.initialize()

    def init_from_value(self, embedding, user_label, asser_label, namelist, asserlist, output_dir="."):
        self.embedding = embedding
        self.user_label = user_label
        self.asser_label = asser_label
        self.namelist = namelist
        self.asserlist = asserlist
        self.output_dir = output_dir
        self.initialize()

    def run_clustering(self, n_clusters=2):
        assert n_clusters == 2

        self.user_clustering_pred, _ = self.k_means(self.embedding[:self.num_user])
        self.asser_clustering_pred, _ = self.k_means(self.embedding[self.num_user:])

        for i in range(self.num_user):
            self.user_clustering_pred[i] = \
                self.user_plot_config[0][0] if self.user_clustering_pred[i] == 0 else self.user_plot_config[1][0]

        for i in range(self.num_asser):
            self.asser_clustering_pred[i] = \
                self.asser_plot_config[0][0] if self.asser_clustering_pred[i] == 0 else self.asser_plot_config[1][0]

    def plot_clustering(self, permulate=None, show=False, save=True, tag=""):
        print("Evaluator plot clustering prediction with config:")
        print("user_plot_config: " + str(self.user_plot_config))
        print("asser_plot_config: " + str(self.asser_plot_config))
        assert self.user_clustering_pred is not None
        assert self.asser_clustering_pred is not None
        pred = np.concatenate([self.user_clustering_pred, self.asser_clustering_pred], axis=0)
        label = np.concatenate([self.user_label, self.asser_label], axis=0)
        # Only plot labeled data
        pred[label == 0] = -1
        pred[label == 1000] = -2
        if self.embedding.shape[1] == 1:
            self.plot_1d(self.embedding, pred, self.user_plot_config, self.asser_plot_config, show, save)
        elif self.embedding.shape[1] == 2:
            self.plot_2d(self.embedding, pred, self.user_plot_config, self.asser_plot_config, show, save)
        elif self.embedding.shape[1] == 3:
            self.plot_3d(self.embedding, pred, self.user_plot_config, self.asser_plot_config, permulate, show, save, tag=tag)

    def dump_topk_json(self, K=50):
        save_path = self.output_dir + "/top_tweet_table.json".format(K)
        res = {"reprA": [], "reprB": [], "reprC": []}

        collection = [(
            self.asserlist[i],
            self.embedding[self.num_user + i][0]
        ) for i in range(self.num_asser)]
        collection = sorted(collection, key=lambda x: x[1], reverse=True)
        for item in collection[:K]:
            res["reprA"].append(item[0])

        collection = [(
            self.asserlist[i],
            self.embedding[self.num_user + i][1]
        ) for i in range(self.num_asser)]
        collection = sorted(collection, key=lambda x: x[1], reverse=True)
        for item in collection[:K]:
            res["reprB"].append(item[0])

        collection = [(
            self.asserlist[i],
            self.embedding[self.num_user + i][2]
        ) for i in range(self.num_asser)]
        collection = sorted(collection, key=lambda x: x[1], reverse=True)
        for item in collection[:K]:
            res["reprC"].append(item[0])

        with open(save_path, "w") as fout:
            json.dump(res, fout, indent=2)

    def plot(self, permulate=None, show=False, save=True, tag=""):
        print("Evaluator plot label with config:")
        print("user_plot_config: " + str(self.user_plot_config))
        print("asser_plot_config: " + str(self.asser_plot_config))
        label = np.concatenate([self.user_label, self.asser_label], axis=0)
        if self.embedding.shape[1] == 1:
            self.plot_1d(self.embedding, label, self.user_plot_config, self.asser_plot_config, show, save)
        elif self.embedding.shape[1] == 2:
            self.plot_2d(self.embedding, label, self.user_plot_config, self.asser_plot_config, show, save)
        elif self.embedding.shape[1] == 3:
            self.plot_3d(self.embedding, label, self.user_plot_config, self.asser_plot_config, permulate, show, save, tag=tag)

    def purity_score(self, y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    def numerical_evaluate(self, verbose=False):
        user_pred = self.user_clustering_pred[self.user_label != 0]
        user_pred_reverse = user_pred.copy()  # 1 --> 2; 2--> 1
        user_pred_reverse = 3 - user_pred_reverse
        user_label = self.user_label[self.user_label != 0]

        asser_pred = self.asser_clustering_pred[self.asser_label != 1000]
        asser_pred_reverse = asser_pred.copy()  # 1001 --> 1002; 1002 --> 1001
        asser_pred_reverse = 2003 - asser_pred_reverse
        asser_label = self.asser_label[self.asser_label != 1000]

        user_pre = [precision_score(user_label, user_pred, pos_label=1),
                    precision_score(user_label, user_pred_reverse, pos_label=1)]
        user_recall = [recall_score(user_label, user_pred, pos_label=1),
                       recall_score(user_label, user_pred_reverse, pos_label=1)]
        user_f1 = [f1_score(user_label, user_pred, pos_label=1),
                   f1_score(user_label, user_pred_reverse, pos_label=1)]

        asser_pre = [precision_score(asser_label, asser_pred, pos_label=1001),
                     precision_score(asser_label, asser_pred_reverse, pos_label=1001)]
        asser_recall = [recall_score(asser_label, asser_pred, pos_label=1001),
                        recall_score(asser_label, asser_pred_reverse, pos_label=1001)]
        asser_f1 = [f1_score(asser_label, asser_pred, pos_label=1001),
                    f1_score(asser_label, asser_pred_reverse, pos_label=1001)]

        log_content = ""
        log_content += "#User: {}, #Assertion: {}".format(self.num_user, self.num_asser) + "\n"
        log_content += "User precision: {:.4f}  User recall: {:.4f}  User F1: {:.4f}".format(user_pre[0], user_recall[0], user_f1[0]) + "\n"
        log_content += "User precision: {:.4f}  User recall: {:.4f}  User F1: {:.4f}".format(user_pre[1], user_recall[1], user_f1[1]) + "\n"
        log_content += "User Purity: {:.4f}".format(self.purity_score(user_label, user_pred)) + "\n"
        log_content += "Assertion precision: {:.4f}  Assertion recall: {:.4f}  Assertion F1: {:.4f}".format(
            asser_pre[0], asser_recall[0], asser_f1[0]) + "\n"
        log_content += "Assertion precision: {:.4f}  Assertion recall: {:.4f}  Assertion F1: {:.4f}".format(
            asser_pre[1], asser_recall[1], asser_f1[1]) + "\n"
        log_content += "Assertion Purity: {:.4f}".format(self.purity_score(asser_label, asser_pred)) + "\n"
        if verbose:
            print(log_content)
        return log_content, max(user_f1[0], user_f1[1]), max(asser_f1[0], asser_f1[1])

    # -------------------------------- Function Utils --------------------------------

    def dbscan(self, embedding, cosine_norm=False, eps=0.5, min_samples=5):
        db = DBSCAN(eps=eps, min_samples=min_samples)
        if cosine_norm:
            length = np.sqrt((embedding ** 2).sum(axis=1))[:, None]
            embedding = embedding / length
        clustering = db.fit(embedding)
        pred = clustering.labels_
        pred[pred < 0] = 0
        return pred, clustering

    def mean_shift(self, embedding, cosine_norm=False, bandwidth=None):
        ms = MeanShift(bandwidth=bandwidth)
        if cosine_norm:
            length = np.sqrt((embedding ** 2).sum(axis=1))[:, None]
            embedding = embedding / length
        clustering = ms.fit(embedding)
        return clustering.labels_, clustering

    def k_means(self, embedding, cosine_norm=False, n_clusters=2, n_init=10):
        km = KMeans(
            n_clusters=n_clusters, n_init=n_init
        )

        if cosine_norm:
            length = np.sqrt((embedding ** 2).sum(axis=1))[:, None]
            embedding = embedding / length

        km_result = km.fit_predict(embedding)
        return km_result, km

    def time_tag(self):
        return time.strftime("%Y%m%d%H%M%S_", time.localtime()) + str(time.time()).split(".")[1]

    def plot_1d(self, embedding, label, user_plot_config, asser_plot_config, show=False, save=True):
        assert embedding.shape[1] == 1
        assert embedding.shape[0] == label.shape[0]

        for l, c, t in user_plot_config:
            emb = embedding[label == l]
            plt.scatter(emb[:, 0].reshape(-1),
                        np.zeros(emb[:, 0].reshape(-1).shape) + 0.15 * np.random.random(
                            size=emb[:, 0].reshape(-1).shape),
                        marker="o", color=c, label=t, s=10)

        for l, c, t in asser_plot_config:
            emb = embedding[label == l]
            plt.scatter(emb[:, 0].reshape(-1),
                        np.ones(emb[:, 0].reshape(-1).shape) + 0.15 * np.random.random(
                            size=emb[:, 0].reshape(-1).shape),
                        marker="^", color=c, label=t, s=10)

        plt.tick_params(labelsize=14)
        plt.legend(loc='best', prop={'size': 14})
        if save:
            plt.savefig(self.output_dir + "/1d_evaluation_{}.jpg".format(self.time_tag()), dpi=500)
        if show:
            plt.show()

    def plot_2d(self, embedding, label, user_plot_config, asser_plot_config, show=False, save=True):
        assert embedding.shape[1] == 2
        assert embedding.shape[0] == label.shape[0]

        for l, c, t in user_plot_config:
            emb = embedding[label == l]
            plt.scatter(emb[:, 0].reshape(-1), emb[:, 1].reshape(-1), marker="o", color=c, label=t, s=10)

        for l, c, t in asser_plot_config:
            emb = embedding[label == l]
            plt.scatter(emb[:, 0].reshape(-1), emb[:, 1].reshape(-1), marker="^", color=c, label=t, s=10)

        plt.tick_params(labelsize=16)
        plt.legend(loc='best', prop={'size': 14})
        if save:
            plt.savefig(self.output_dir + "/2d_evaluation_{}.jpg".format(self.time_tag()), dpi=500)
        if show:
            plt.show()

    def plot_3d(self, embedding, label, user_plot_config, asser_plot_config, permulate=None, show=False, save=True, tag=""):
        if permulate is None:
            permulate = [0, 1, 2]
        assert embedding.shape[1] == 3
        assert embedding.shape[0] == label.shape[0]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        for l, c, t in user_plot_config:
            emb = embedding[label == l]
            ax.scatter(emb[:, permulate[0]].reshape(-1), emb[:, permulate[1]].reshape(-1), emb[:, permulate[2]].reshape(-1),
                       marker="o", color=c, label=t)

        for l, c, t in asser_plot_config:
            emb = embedding[label == l]
            ax.scatter(emb[:, permulate[0]].reshape(-1), emb[:, permulate[1]].reshape(-1), emb[:, permulate[2]].reshape(-1),
                       marker="^", color=c, label=t)

        plt.legend(loc='upper right', prop={'size': 14})
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_zlabel("Dim 3")
        ax.view_init(20, 120)

        if save:
            plt.savefig(self.output_dir + "/3d_evaluation_cluster_{}_{}.pdf".format(tag, self.time_tag()), bbox_inches='tight')
        if show:
            plt.show()

if __name__ == "__main__":
    # For Election
    # evaluator = Evaluator(
    #     user_plot_config=[
    #             [2, "#178bff", "User View 1"],
    #             # [0, "#3b3b3b", "User Neu"],
    #             [1, "#ff5c1c", "User View 2"]
    #         ],
    #     asser_plot_config=[
    #             [2, "#30a5ff", "Tweet View 1"],
    #             # [0, "#4d4d4d", "Assertion Neu"],
    #             [1, "#fc8128", "Tweet View 2"]
    #         ],
    #     use_b_matrix=False
    # )

    # For Voteview
    evaluator = Evaluator(
        user_plot_config=[
                [2, "#178bff", "Congressman View 1"],
                # [0, "#3b3b3b", "User Neu"],
                [1, "#ff5c1c", "Congressman View 2"]
            ],
        asser_plot_config=[
                [2, "#30a5ff", "Bill View 1"],
                # [0, "#4d4d4d", "Assertion Neu"],
                [1, "#fc8128", "Bill View 2"]
            ],
        use_b_matrix=False
    )

    evaluator.init_from_dir(
        "output/InfoVGAE_bill_3D_20210525153347"
    )

    evaluator.plot(show=True)
    evaluator.run_clustering()
    evaluator.plot_clustering(show=True)
    evaluator.numerical_evaluate()
    evaluator.dump_topk_json()
