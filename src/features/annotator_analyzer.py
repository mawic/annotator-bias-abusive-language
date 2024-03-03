import random

import krippendorff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import src.data.data_loader as dl
import src.features.aggregation_helper as ha
import src.features.annotation_aggregator as aa
import src.features.cluster as clu
import src.features.graph_generation as gg
import src.features.helper as he
import src.features.krippendorffs_alpha as ka
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from src.features.annotation_aggregator import DawidSkeneAggregator
from tqdm import tqdm


def get_ira_per_worker(annotations_wide, gold_labels):
    worker_ids = []
    ira_scores = []
    for worker_id in tqdm(annotations_wide.columns[2:]):
        worker_ids.append(worker_id)

        annotations_of_worker = annotations_wide.loc[:, ["id", worker_id]].dropna()
        annotations_of_worker = annotations_of_worker.sort_values(by="id")
        id_workers_annotations = annotations_of_worker["id"].to_list()
        labels_workers_annotations = annotations_of_worker[worker_id].to_list()

        relevant_gold_documents = gold_labels[
            gold_labels["id"].isin(id_workers_annotations)
        ].sort_values(by="id")
        relevant_gold_labels = relevant_gold_documents["label"].to_list()

        labels_workers_annotations = [
            np.nan if np.isnan(v) else int(v) for v in labels_workers_annotations
        ]
        relevant_gold_labels = [
            np.nan if np.isnan(v) else int(v) for v in relevant_gold_labels
        ]

        reliability_data = [relevant_gold_labels, labels_workers_annotations]
        ira_scores.append(
            krippendorff.alpha(
                reliability_data=reliability_data, level_of_measurement="nominal"
            )
        )
    ira_scores = np.asarray(ira_scores)
    return worker_ids, ira_scores


def compareLabels(list_1, list_2):
    equal = 0
    if len(list_1) != len(list_2):
        return False
    for i, j in zip(list_1, list_2):
        if i == j:
            equal += 1
    equal_rel = equal / len(list_1)
    print("Agreement:\t", equal, f"\t{equal_rel:.2%}")
    print("Disgreement:\t", len(list_1) - equal, f"\t{(1-equal_rel):.2%}")
    print("Total:\t\t", len(list_1))


def getAgreement(list_1, list_2):
    equal = 0
    if len(list_1) != len(list_2):
        return False
    for i, j in zip(list_1, list_2):
        if i == j:
            equal += 1
    equal_rel = equal / len(list_1)
    return equal_rel


class AnnotatorAnalyzer:
    def __init__(self):
        self.x = "Hello"
        self.binary = True
        self.seed = 42

    def set_dataset(self, dataset):
        self.dataset = dataset

    def prepare_data(self, show_stats=True):
        self.load_expert_data(show_stats)
        self.load_annotator_data(show_stats)
        self.assure_expert_amateur_compatability()

    def load_expert_data(self, show_stats=True):
        self.expert_data = self.dataset.get_data(expert_only=True, binary=self.binary)

        expert_data_abs = self.expert_data["label"].value_counts()
        expert_data_rel = self.expert_data["label"].value_counts(normalize=True)
        
        if show_stats:
            # Visualize class distribution of expert labels
            print(expert_data_abs)
            print(expert_data_rel)
            print("Number of documents:\t", len(self.expert_data))

        self.classes = list(expert_data_abs.index)
        self.mapping = dict()
        for i, c in enumerate(self.classes):
            self.mapping[c] = float(i)

        self.expert_data = self.expert_data.sort_values("id").reset_index()
        self.expert_data = self.expert_data.replace({"label": self.mapping})

    def load_annotator_data(self, show_stats=True):
        self.annotator_data_wide = self.dataset.get_data(
            expert_only=False, binary=self.binary, format="wide"
        )
        self.annotator_data_wide = self.annotator_data_wide.sort_values(
            "id"
        ).reset_index()
        for col in self.annotator_data_wide.columns[2:]:
            self.annotator_data_wide = self.annotator_data_wide.replace(
                {col: self.mapping}
            )
        self.annotator_data_wide.columns.name = None
        if 'index' in self.annotator_data_wide.columns:
            self.annotator_data_wide = self.annotator_data_wide.drop(['index'], axis=1)
        
    def assure_expert_amateur_compatability(self, show_stats=True):
        self.expert_data = self.expert_data[
            self.expert_data["id"].isin(self.annotator_data_wide["id"].to_list())
        ]
        self.annotator_data_wide = self.annotator_data_wide[
            self.annotator_data_wide["id"].isin(self.expert_data["id"].to_list())
        ]
        if show_stats:
            print(
                "Expert:",
                len(self.expert_data),
                "Annotators:",
                len(self.annotator_data_wide),
            )

    def get_interrater_reliability_of_workers(self):
        return get_ira_per_worker(self.annotator_data_wide, self.expert_data)

    def aggregate_david_skene(self, tol=0.00001, max_iter=100):
        self.ds = DawidSkeneAggregator()
        self.ds.aggregate(self.annotator_data_wide, tol=tol, max_iter=max_iter)
        self.ds_aggregations = self.ds.get_aggregation(kind="df").reset_index()

        # Expert vs. David Skene
        print("Comparison: Expert vs. David & Skene labels")
        compareLabels(
            self.expert_data.sort_values(by="id")["label"].to_list(),
            self.ds_aggregations.sort_values(by="index")["label"].to_list(),
        )

    def compute_distances(self, ord="fro"):
        if ord == "jensenshannon":
            self.distance_mat = aa.compute_distance_matrix(self.ds)
        else:
            self.distance_mat = aa.compute_distance_matrix(self.ds, ord=ord)

    def plot_distance_matrix(self):
        aa.plot_distance_matrix(self.distance_mat, self.ds)

    def cluster_annotators(self, n_clusters=3):
        # Flatten DS bias matrix
        self.ds_bias_matrices = self.ds.error_rates
        vec_length = self.ds_bias_matrices.shape[1] * self.ds_bias_matrices.shape[2]
        self.ds_bias_flattend = self.ds_bias_matrices.reshape(-1, vec_length)

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.seed)
        self.ds_clusters = kmeans.fit_predict(self.ds_bias_flattend)

        self.ds_group_list = clu.group_annotators_by_cluster(self.ds_clusters)

    def plot_clustered_annotators(self):
        markers = {0: "D", 1: "o", 2: "v", 3: "<", 4: ">", 5: "d"}

        fig, ax = plt.subplots(figsize=(8, 8))
        self.ds_transformed_df = pd.DataFrame(
            self.ds_bias_flattend[:, 1:3], columns=["optimistic", "pessimistic"]
        )
        self.ds_transformed_df["cluster"] = self.ds_clusters

        max_axes = (
            max(
                [
                    max(self.ds_bias_flattend[:, 2:3].flatten()),
                    max(self.ds_bias_flattend[:, 1:2].flatten()),
                ]
            )
            * 1.1
        )
        plt.ylim(-0.02, max_axes)
        plt.xlim(-0.02, max_axes)

        return sns.scatterplot(
            ax=ax,
            data=self.ds_transformed_df,
            x="optimistic",
            y="pessimistic",
            style="cluster",
            hue="cluster",
            markers=markers,
            legend=False,
            s=100,
        )

    def simulate_annotation_strategies(self, distance_types=["fro"],exclude_from_plot=[],distance_ascending=True):
        if len(distance_types) == 0:
            return False
        
        annotator_data_wide = self.annotator_data_wide.copy()
        data = []
        for distance_type in distance_types:
            self.compute_distances(ord=distance_type)

            # add distance to annotation data
            annotator_data_wide["distance"] = -1
            annotator_data_wide["agreement"] = False
            for index, row in annotator_data_wide.iterrows():
                # remove NaN valeus from list
                ls = np.array(row[2:-2].values, dtype=np.float64)
                not_nan = ls[~np.isnan(ls)]

                if len(not_nan) == 2 and len(set(not_nan)) == 1:
                    # annotators agree on annotation
                    annotator_data_wide.loc[index, "agreement"] = True
                positions = np.argwhere(~np.isnan(ls)).flatten()
                if len(positions) == 2:
                    # annotators disagree on annotation
                    # add distance
                    annotator_data_wide.loc[index, "distance"] = self.distance_mat[
                        positions[0]
                    ][positions[1]]

            # combine data
            df_tmp = pd.merge(
                self.expert_data,
                self.ds_aggregations,
                how="left",
                left_on="id",
                right_on="index",
            )
            df_tmp = pd.merge(
                df_tmp,
                annotator_data_wide[["id", "distance", "agreement"]],
                how="left",
                left_on="id",
                right_on="id",
            )
            
            # label_y = david & skene labels
            # label_x = gold standard
            df_tmp["ds_random"] = df_tmp["label_y"]
            df_tmp["majority_vote"] = ""
            df_tmp["majority_vote_random"] = ""
            for index, row in df_tmp.iterrows():
                if row["agreement"]:
                    # overwrite ds label / ds sometimes flipls the labels
                    df_tmp.loc[index, "label_y"] = row["label_x"]
                    df_tmp.loc[index, "ds_random"] = row["label_x"]
                    df_tmp.loc[index, "majority_vote"] = row["label_x"]
                    df_tmp.loc[index, "majority_vote_random"] = row["label_x"]
                else:
                    df_tmp.loc[index, "majority_vote"] = -1
                    df_tmp.loc[index, "majority_vote_random"] = float(random.randint(0, 1))

            # simulate expert annotations
            no_disagreement = len(
                annotator_data_wide[annotator_data_wide["agreement"] == False]
            )

            df_tmp = df_tmp.sort_values(
                by=["agreement", "distance"], ascending=[True, distance_ascending]
            )
            for i in tqdm(range(0, no_disagreement)):
                new_ds = df_tmp["label_x"].to_list()[:i] + df_tmp["label_y"].to_list()[i:]
                data.append([i, getAgreement(df_tmp["label_x"].to_list(), new_ds), f"ds_distance_{distance_type}"])

        
        # add baselines
        df_tmp = df_tmp.sample(frac=1)
        df_tmp = df_tmp.sort_values(by=["agreement"], ascending=[True])        
        for i in range(0, no_disagreement):
            # david & skene random annotation
            new_ds = df_tmp["label_x"].to_list()[:i] + df_tmp["ds_random"].to_list()[i:]
            data.append(
                [i, getAgreement(df_tmp["label_x"].to_list(), new_ds), "ds_random"]
            )     

            # majority vote  annotations
            new_mv = (
                df_tmp["label_x"].to_list()[:i] + df_tmp["majority_vote"].to_list()[i:]
            )
            data.append(
                [i, getAgreement(df_tmp["label_x"].to_list(), new_mv), "majority_vote"]
            )

            # majority vote random annotations
            new_mv_rnd = (
                df_tmp["label_x"].to_list()[:i]
                + df_tmp["majority_vote_random"].to_list()[i:]
            )
            data.append(
                [
                    i,
                    getAgreement(df_tmp["label_x"].to_list(), new_mv_rnd),
                    "majority_vote_random",
                ]
            )
            
            
        # combine all results in one dataframe   
        df = pd.DataFrame(data, columns=["steps", "rate", "type"])

        fig, ax = plt.subplots(figsize=(12,12))
        plot = sns.lineplot(ax=ax,data=df[~df['type'].isin(exclude_from_plot)], x="steps", y="rate", hue="type")