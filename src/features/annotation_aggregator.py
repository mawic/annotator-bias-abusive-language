import sys

sys.path.append("../")
import os
import pathlib
import pickle
import statistics
import subprocess
import uuid
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import src.features.dawid_skene as ds
from scipy import stats
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, cohen_kappa_score
from tqdm import tqdm

sns.set_theme()
from definitions import PROJECT_DIR


def compare_aggregators(aggregators, eval_func="accuracy", kind="array"):
    aggregator_names = [type(agg).__name__.replace("Aggregator", "") for agg in aggregators]
    results = np.zeros((len(aggregators), len(aggregators)))
    for i in range(len(aggregators)):
        agg1 = aggregators[i]
        for j in range(len(aggregators)):
            agg2 = aggregators[j]
            results[i,j] = agg1.evaluate(agg2.get_aggregation(), kind=eval_func)
    if kind == "accuracy":
        return results
    if kind == "df":
        return pd.DataFrame(results, index=aggregator_names, columns=aggregator_names)
    return results

def combine_aggregators(aggregators_dict, join="outer"):
    aggregation_dfs = []
    names = []
    for key, aggregator in aggregators_dict.items():
        aggregation_dfs.append(aggregator.get_aggregation(kind="df"))
        names.append(key)
    result = pd.concat(aggregation_dfs, axis=1, sort=False, join=join).reset_index()
    result.columns = ["index", *names]
    return result

def run_aggregations(annotations_dict, kind):
    names = annotations_dict.keys()
    print(f'Building {len(names)} seperate aggregators for {names} annotations')

    aggregators_dict = {}
    for key, annotations in annotations_dict.items():
        if kind == "mj":
            agg = MajorityVoteAggregator()
        elif kind == "ds":
            agg = DawidSkeneAggregator()
        elif kind == "mace":
            agg = MaceAggregator()
        else:
            print(f'Aggregator {kind} not supported')
            return None
        print(f'Running {type(agg).__name__} on {key} annotations')
        agg.aggregate(annotations)
        aggregators_dict[key] = agg
    return aggregators_dict

def aggregate_per_cluster(data_wide, cluster_annotator_names, kind):
    """
    Splits the data frame according to the clusters and
    performs the aggregation as specified.
    """
    data_wide_dict = {}
    data_wide_dict["full"] = data_wide
    for cluster_id, annotator_names in enumerate(cluster_annotator_names):
        columns_to_select = ["text", *annotator_names]
        selected_df = data_wide[columns_to_select]
        selected_df = selected_df.dropna(thresh=2) # Threshold 2 only if text is in the columns
        data_wide_dict[cluster_id] = selected_df
    aggregators_dict = run_aggregations(data_wide_dict, kind=kind)
    labels_dict = {key: value.get_aggregation(kind="df") for (key, value) in aggregators_dict.items()}
    labels_dict = {key: value.rename(columns={"label": f"{key}_label"}) for (key, value) in labels_dict.items()}
    labels_cluster_df = pd.concat([value for (key, value) in labels_dict.items()], axis=1)
    labels_cluster_df = labels_cluster_df.loc[:, ~labels_cluster_df.columns.duplicated()]
    return labels_cluster_df

def load(model_name):
    model_path = "../../models/aggregators"
    complete_path = os.path.join(pathlib.Path(__file__).parent.absolute(), model_path, model_name)
    pck_file = open(complete_path , "rb" )
    model = pickle.load(pck_file)
    pck_file.close()
    print(f"Loaded {model_name}")
    return model

def _plot_error_matrices(error_rates, observers, classes):

    for i, obs in enumerate(observers):
        error_rate = error_rates[i]

        ax = sns.heatmap(error_rate, vmin=0, vmax=1, annot=True, cbar=False)
        ax.set_title(f'{obs}', {'fontsize': 'large', 'fontweight': 'bold'})
        ax.set_xlabel('Observed label')
        ax.set_ylabel('Latent truth')
        ax.set_xticklabels(classes, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(classes, rotation=0, ha="right", rotation_mode="anchor")

        plt.show(ax)

def plot_bias_matrices(aa):
    obs = aa.worker_mapping
    bias = aa.error_rates
    classes = aa.class_mapping
    _plot_error_matrices(bias, obs, classes)

def compute_distance_matrix(aa, ord="fro"):
    num_annotators = len(aa.worker_mapping)
    bias_matrices = aa.error_rates
    dist_mat = np.zeros((num_annotators, num_annotators))

    for i in tqdm(range(num_annotators)):
        for j in range(num_annotators):
            mat_i = bias_matrices[i]
            mat_j = bias_matrices[j]
            mat_diff = mat_i - mat_j
            mat_distance = np.linalg.norm(mat_diff, ord=ord)
            dist_mat[i,j] = mat_distance
    return dist_mat

def compute_distance_matrix_2(annotators,matrices,ord="fro"):
    num_annotators = len(annotators)
    bias_matrices = matrices
    dist_mat = np.zeros((num_annotators, num_annotators))

    for i in tqdm(range(num_annotators)):
        for j in range(num_annotators):
            mat_i = bias_matrices[i]
            mat_j = bias_matrices[j]
            mat_diff = mat_i - mat_j
            mat_distance = np.linalg.norm(mat_diff, ord=ord)
            dist_mat[i,j] = mat_distance
    return dist_mat

def compute_jensenshannon_distance_matrix(aa):
    num_annotators = len(aa.worker_mapping)
    bias_matrices = aa.error_rates
    dist_mat = np.zeros((num_annotators, num_annotators))

    for i in tqdm(range(num_annotators)):
        for j in range(num_annotators):
            vect_i = bias_matrices[i].reshape(-1)/2
            vect_j = bias_matrices[j].reshape(-1)/2
            mat_distance = distance.jensenshannon(vect_i,vect_j)
            dist_mat[i,j] = mat_distance
    return dist_mat

def plot_distance_matrix(distance_matrix, ds):
    mask = np.zeros_like(distance_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    mask[np.diag_indices_from(mask)] = False

    fig, ax = plt.subplots(figsize=(20,15))
    ax = sns.heatmap(distance_matrix, annot=True, ax=ax, mask=mask)
    ax.set_title(f"Distance between annotators' bias matrices", {'fontsize': 'large', 'fontweight': 'bold'})
    ax.set_xticklabels(ds.worker_mapping, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(ds.worker_mapping, rotation=0, ha="right", rotation_mode="anchor")
    plt.show(ax)
    


def plot_distance_matrix_2(distance_matrix, annotators):
    mask = np.zeros_like(distance_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    mask[np.diag_indices_from(mask)] = False

    fig, ax = plt.subplots(figsize=(20,15))
    ax = sns.heatmap(distance_matrix, annot=True, ax=ax, mask=mask)
    ax.set_title(f"Distance between annotators' bias matrices", {'fontsize': 'large', 'fontweight': 'bold'})
    ax.set_xticklabels(annotators, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(annotators, rotation=0, ha="right", rotation_mode="anchor")
    plt.show(ax)

def aggregate_bias_matrices(bias_matrices, classes=[], group_list=[]):
    num_annotators = len(bias_matrices)
    if not group_list:
        # Aggregate matrix for all annotators
        bias_agg = np.sum(bias_matrices, axis=0) / len(bias_matrices)
        ax = sns.heatmap(bias_agg, vmin=0, vmax=1, annot=True, cbar=False)
        ax.set_title(f'All annotators (n = {int(num_annotators)})', {'fontsize': 'large', 'fontweight': 'bold'})
        ax.set_xlabel('Observed label')
        ax.set_ylabel('Latent truth')

        if classes:
            ax.set_xticklabels(classes, rotation=45, ha="right", rotation_mode="anchor")
            ax.set_yticklabels(classes, rotation=0, ha="right", rotation_mode="anchor")
        plt.show(ax)
    else:
        for id, annotator_ids in enumerate(group_list):
            selected_bias_matrices = bias_matrices[annotator_ids]
            bias_agg = np.sum(selected_bias_matrices, axis=0) / len(selected_bias_matrices)
            ax = sns.heatmap(bias_agg, vmin=0, vmax=1, annot=True, cbar=False)
            ax.set_title(f'Cluster  (n = {len(annotator_ids)})', {'fontsize': 'large', 'fontweight': 'bold'})
            ax.set_xlabel('Observed label')
            ax.set_ylabel('Latent truth')

            if classes:
                ax.set_xticklabels(classes, rotation=45, ha="right", rotation_mode="anchor")
                ax.set_yticklabels(classes, rotation=0, ha="right", rotation_mode="anchor")
            plt.show(ax)

class AnnotationAggregator:

    def _prepare_text_data(self, annotations):
        self.text = annotations["text"]
        annotations = annotations.drop(columns="text")
        return annotations

        # annotations = annotations[[]]

    def _prepare_wide_data(self, annotations):
        """
        Annotations is the input as a Pandas DataFrame.
        Each row must represent a comment and each column one annotator.
        The values are the annotations.
        """
        if "text" in annotations.columns:
            annotations = self._prepare_text_data(annotations)

        if "id" not in annotations.columns:
            annotations = annotations.reset_index()

        self.comment_mapping = annotations["id"].values
        self.worker_mapping = annotations.columns[1:].values
        unique_classes = pd.unique(annotations.iloc[:, 1:].values.ravel('K'))
        self.class_mapping = sorted(unique_classes[~pd.isnull(unique_classes)])
        self._print_data_info()
        return annotations


    def _prepare_long_data(self, annotations):
        if "id" not in annotations.columns:
            annotations = annotations.reset_index()

        if "text" in annotations.columns:
            annotations = self._prepare_text_data(annotations)

        self.comment_mapping = annotations["id"].unique()
        self.worker_mapping = annotations["annotator"].unique()
        self.class_mapping = sorted(annotations["label"].unique())
        self._print_data_info()
        return annotations


    def _print_data_info(self):
        self.num_comments = len(self.comment_mapping)
        self.num_workers = len(self.worker_mapping)
        self.num_classes = len(self.class_mapping)

        print('----------------------------------')
        print(f'{self.num_comments} comments')
        print(f'{self.num_workers} annotators')
        print(f'{self.num_classes} classes: {self.class_mapping}')
        print('----------------------------------')


    def aggregate(self, annotations):
        """
        Annotations is the input as a Pandas DataFrame.
        Each row must represent a comment and each column one annotator.
        The values are the annotations.
        """
        pass

    def get_aggregation(self, kind="array", return_text=True):
        if kind == "array":
            return self.aggregation
        if kind == "df":
            df = pd.DataFrame(self.aggregation, index=self.comment_mapping, columns=["label"])
            if return_text and len(self.text) > 0:
                df["text"] = self.text.values
                df = df[["text", "label"]]
            return df
        if kind == "plot":
            df = pd.DataFrame(self.aggregation, index=self.comment_mapping, columns=["label"])
            counts = df["label"].value_counts()
            ax = sns.barplot(x=counts.index, y=counts)
            ax.set_xlabel('Aggregated class labels')
            ax.set_ylabel('Counts')
            return ax
        if kind == "summary":
            desc = stats.describe(self.aggregation)._asdict()
            df = pd.DataFrame([desc], columns=desc.keys())
            return df

    def get_distribution(self, kind="array"):
        if kind == "array":
            return self.distribution
        if kind == "df":
            return pd.DataFrame(self.distribution, index=self.comment_mapping, columns=self.class_mapping)


    def get_worker_reliability(self, kind="array"):
        if kind == "array":
            return self.worker_reliability
        if kind == "df":
            return pd.DataFrame(self.worker_reliability, index=self.worker_mapping, columns=["reliability"])
        if kind == "plot":
            df = pd.DataFrame(self.worker_reliability, index=self.worker_mapping, columns=["reliability"])
            ax = sns.histplot(df, x="reliability")
            ax.set_xlabel('Reliability')
            return ax

    def evaluate(self, test_labels, kind="accuracy"):
        # Add confusion matrix and return as plot
        # Pass test object instead of label list
        assert(len(self.aggregation) == len(test_labels))
        if kind == "accuracy":
            return accuracy_score(self.aggregation, test_labels)
        if kind == "kappa":
            return round(cohen_kappa_score(self.aggregation, test_labels), 2)

    def save(self):
        model_path = "../../models/aggregators"
        model_id = str(uuid.uuid4())
        model_name = f'{model_id}_{type(self).__name__}.pck'
        complete_path = os.path.join(pathlib.Path(__file__).parent.absolute(), model_path, model_name)
        pck_file = open(complete_path , "wb" )
        pickle.dump(self, pck_file)
        pck_file.close()
        print(f"Saved {type(self).__name__} in {complete_path}")
        return model_name



    # def evaluate_distribution kind=kl or kind=js??

class MajorityVoteAggregator(AnnotationAggregator):

    def _prepare_data(self, annotations):
        if "annotator" in annotations.columns:
            annotations = super()._prepare_long_data(annotations)
            return annotations
        else:
            annotations = super()._prepare_wide_data(annotations)
            melt_df = annotations.melt(id_vars="id").rename(columns={'variable': 'annotator', 'value': 'label'})
            melt_df = melt_df.dropna()
            return melt_df

    def aggregate(self, annotations):
        melt_df = self._prepare_data(annotations)
        print("Running majority voting")

        start = datetime.now()
        agg_df = melt_df.groupby(["id"]).agg(aggregation=pd.NamedAgg(column='label', aggfunc=lambda x: max(x.tolist(), key=x.tolist().count)))
        self.aggregation = agg_df.values.flatten()
        end = datetime.now()

        print(f'Completed majority_voting in {end-start} seconds')
        print()
        return None


class DawidSkeneAggregator(AnnotationAggregator):

    def _prepare_data(self, annotations):
        if "annotator" in annotations.columns:
            annotations = super()._prepare_long_data(annotations)
        else:
            annotations = super()._prepare_wide_data(annotations)
            melt_df = annotations.melt(id_vars="id").rename(columns={'variable': 'annotator', 'value': 'label'})
            melt_df = melt_df.dropna()
            annotations =  melt_df

        data = {}
        data_split = annotations.to_dict('split')['data']
        for data_point in data_split:
            comment = data_point[0]
            worker = data_point[1]
            annotation = data_point[2]
            if comment not in data:
                data[comment] = {}
            if worker not in data[comment]:
                data[comment][worker] = []
            data[comment][worker].append(annotation)
        return data

    def aggregate(self, annotations, tol=0.00001, max_iter=100):
        print("Preparing data")
        data = self._prepare_data(annotations)


        print("Running Dawid&Skene")
        start = datetime.now()
        comments, workers, classes, counts, class_marginals, error_rates, incidence_of_error_rates, comment_classes = ds.run(data, tol=tol, max_iter=max_iter, print_results=False, verbose=False)

        self.error_rates = error_rates
        self.distribution = comment_classes
        class_lookup = np.broadcast_to(self.class_mapping, (len(self.distribution), len(self.class_mapping)))
        self.aggregation = np.take(class_lookup, np.argmax(self.distribution, axis=1))

        self.worker_confusion = incidence_of_error_rates
        incidence_trace = np.trace(incidence_of_error_rates, axis1=1, axis2=2)
        incidence_optimism_score = np.triu(incidence_of_error_rates, k=1).sum(axis=(1,2))
        incidence_pessimism_score = np.tril(incidence_of_error_rates, k=-1).sum(axis=(1,2))

        self.worker_reliability = incidence_trace
        self.worker_scores = np.stack((incidence_trace, incidence_optimism_score, incidence_pessimism_score), axis=-1)
        end = datetime.now()
        print(f'Completed Dawid&Skene in {end-start} seconds')
        print()
        return None

    def get_worker_scores(self, kind="array"):
        if kind == "array":
            return self.worker_scores
        if kind == "df":
            return pd.DataFrame(self.worker_scores, index=self.worker_mapping, columns=["reliability", "optimism", "pessimism"])
        if kind == "plot":
            df = pd.DataFrame(self.worker_scores, index=self.worker_mapping, columns=["reliability", "optimism", "pessimism"])
            fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
            sns.scatterplot(data=df, x="reliability", y="optimism", ax=ax1)
            sns.scatterplot(data=df, x="reliability", y="pessimism", ax=ax2)
            sns.scatterplot(data=df, x="optimism", y="pessimism", ax=ax3)
            plt.show(fig.tight_layout())
            #
        # Performance issues when plotting
        #     df = pd.DataFrame(self.worker_scores, index=self.worker_mapping, columns=["reliability", "optimisim", "pessimism"])
        #     fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True)
        #     sns.histplot(df, x="reliability", ax=ax1)
        #     ax1.set_xlabel('Reliability')
        #     sns.histplot(df, x="optimisim", ax=ax2)
        #     ax2.set_xlabel('Optimisim')
        #     sns.histplot(df,  x="pessimism", ax=ax3)
        #     ax3.set_xlabel('Pessimism')
        #     return fig

    def get_worker_confusion_matrix():
        if kind == "array":
            return self.worker_confusion
        if kind == "plot":
            pass
        # Plot confusion matrixes


class MaceAggregator(AnnotationAggregator):

    def __init__(self):
        super().__init__()
        self.MACE_PATH = os.path.join(PROJECT_DIR, "../MACE-master/MACE")
        self.MACE_TEMP_PATH = os.path.join(pathlib.Path(__file__).parent.absolute(), 'MACE_TEMP')

    def _prepare_data(self, annotations):
        annotations = super()._prepare_wide_data(annotations)
        # Make id to index for Mace
        annotations = annotations.set_index("id")
        self.mace_id = str(uuid.uuid4())
        file_name = f'{self.mace_id}_MACE_input.csv'
        temp_path = os.path.join(self.MACE_TEMP_PATH, file_name)
        annotations.to_csv(temp_path, header=False, index=False)
        return temp_path

    def aggregate(self, annotations):
        input_path = self._prepare_data(annotations)

        print("Running MACE")
        start = datetime.now()
        completed_process = subprocess.run([self.MACE_PATH, '--distribution', f'--prefix {self.mace_id}_MACE_output', input_path],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print(completed_process.stderr)

        prediction_path = os.path.join(os.getcwd(), f'{self.mace_id}_MACE_output.prediction')
        competence_path = os.path.join(os.getcwd(), f'{self.mace_id}_MACE_output.competence')
        predictions_df = pd.read_table(prediction_path, header=None)
        predictions_df = predictions_df.applymap(lambda x: tuple(x.split(" ")))
        predictions_df = predictions_df.apply(lambda x: sorted(x.to_list()), axis=1, result_type='expand')
        predictions_df = predictions_df.applymap(lambda x: float(x[1]))
        self.distribution = predictions_df.values


        class_lookup = np.broadcast_to(self.class_mapping, (len(self.distribution), len(self.class_mapping)))
        self.aggregation = np.take(class_lookup, np.argmax(self.distribution, axis=1))

        competence_df = pd.read_table(competence_path, header=None)
        self.worker_reliability = competence_df.values.flatten()

        print("Deleting temporary files:")
        os.remove(input_path)
        os.remove(prediction_path)
        os.remove(competence_path)
        print(f"{input_path}")
        print(f"{prediction_path}")
        print(f"{competence_path}")
        print()
        end = datetime.now()
        print(f'Completed MACE in {end-start} seconds')
        print()
        return None
