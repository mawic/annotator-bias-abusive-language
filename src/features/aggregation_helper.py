from sklearn.metrics import confusion_matrix
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

def plot_bias_matrices(error_rates, observers, classes):
    for i, obs in enumerate(observers):
        error_rate = error_rates[i]

        ax = sns.heatmap(error_rate, vmin=0, vmax=1, annot=True, cbar=False)
        ax.set_title(f'{obs}', {'fontsize': 'large', 'fontweight': 'bold'})
        ax.set_xlabel('Observed labels')
        ax.set_ylabel('Gold labels')
        ax.set_xticklabels(classes, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(classes, rotation=0, ha="right", rotation_mode="anchor")
        plt.show(ax)

def compute_distance_matrix(error_rates, observers):
    num_annotators = len(observers)
    bias_matrices = error_rates
    dist_mat = np.zeros((num_annotators, num_annotators))

    for i in tqdm(range(num_annotators)):
        for j in range(num_annotators):
            mat_i = bias_matrices[i]
            mat_j = bias_matrices[j]
            mat_diff = mat_i - mat_j
            mat_distance = np.linalg.norm(mat_diff)
            dist_mat[i,j] = mat_distance
    return dist_mat

def plot_distance_matrix(distance_matrix, observers):
    mask = np.zeros_like(distance_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    mask[np.diag_indices_from(mask)] = False

    fig, ax = plt.subplots(figsize=(20,15))
    ax = sns.heatmap(distance_matrix, annot=True, ax=ax, mask=mask)
    ax.set_title(f"Distance between annotators' bias matrices", {'fontsize': 'large', 'fontweight': 'bold'})
    ax.set_xticklabels(observers, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(observers, rotation=0, ha="right", rotation_mode="anchor")
    plt.show(ax)

def get_bias_matrix_per_annotator(annotations_wide, gold_labels, classes, normalize="true"):
    worker_ids = []
    confusion_matrices = []
    for worker_id in annotations_wide.columns[2:]:
        worker_annotations_df = annotations_wide.loc[:, ["id", worker_id]].dropna()
        annotated_revs = worker_annotations_df["id"]
        worker_annotations = worker_annotations_df[worker_id].values
        relevant_gold_labels = gold_labels[gold_labels['id'].isin(annotated_revs)]['label'].to_list()
        conf = confusion_matrix(relevant_gold_labels, worker_annotations, labels=classes, normalize=normalize)
        worker_ids.append(worker_id)
        confusion_matrices.append(conf)
    confusion_matrices = np.asarray(confusion_matrices)
    return worker_ids, confusion_matrices    
    
    
    
def get_confusion_per_worker(annotations_wide, gold_labels, classes, normalize="true"):
    """
    Computes a confusion matrix for all workers in the columns of annotations_wide respective to the gold label

    Returns list of worker_ids, list of confusion matrix of the same length
    """
    worker_ids = []
    confusion_matrices = []
    for worker_id in tqdm(annotations_wide.columns[1:]):
        worker_annotations_df = annotations_wide.loc[:, ["rev_id", worker_id]].dropna()
        annotated_revs = worker_annotations_df["rev_id"]
        worker_annotations = worker_annotations_df[worker_id].values
        relevant_gold_labels = gold_labels.iloc[gold_labels.index.isin(annotated_revs)].values.flatten()
        conf = confusion_matrix(relevant_gold_labels, worker_annotations, labels=classes, normalize=normalize)
        worker_ids.append(worker_id)
        confusion_matrices.append(conf)
    confusion_matrices = np.asarray(confusion_matrices)
    return worker_ids, confusion_matrices


def get_confusion_scores_per_worker(worker_id_list, confusion_matrix_list):
    num_comments = confusion_matrix_list.sum(axis=(1,2))
    num_comments_mapped = num_comments[..., None, None]
    conf_normalized = confusion_matrix_list / num_comments_mapped

    reliability = np.trace(conf_normalized, axis1=1, axis2=2)
    optimism = np.triu(conf_normalized, k=1).sum(axis=(1,2))
    pessimism = np.tril(conf_normalized, k=-1).sum(axis=(1,2))
    num_comments = num_comments.flatten()

    data = {"worker_id": worker_id_list, "reliability": reliability, "optimism": optimism, "pessimism": pessimism, "annotations": num_comments}
    df = pd.DataFrame(data)
    return df


def aggregate_confusion_scores(confusion_df, weighted=False):
    if weighted:
        wm = lambda x: print(x.shape, confusion_df.loc[x.index, "annotations"].shape) #np.average(x, weights=confusion_df.loc[x.index, "annotations"])
        agg_df = confusion_df.groupby("group").agg(wm)
    else:
        agg_df = confusion_df.groupby("group").agg("mean")
    return agg_df[["reliability", "optimisim", "pessimism"]]
