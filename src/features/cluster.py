import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


def evalutate_clusters(gold_labels, data_wide, annotator_names_clusters, plot=False):
    data_wide = data_wide.set_index("id")

    for cluster_id, annotator_names in enumerate(annotator_names_clusters):
        annotator_value_counts = []
        gold_value_counts = []
        for annotator_name in annotator_names:
            annotator_annotations = data_wide[[annotator_name]].dropna(how="all")
            annotator_value_count = annotator_annotations.value_counts(normalize=True)
            gold_labels_selected = gold_labels[gold_labels["id"].isin(annotator_annotations.index)]
            gold_labels_sel_value_count = gold_labels_selected["label"].value_counts(normalize=True)
            annotator_value_counts.append(annotator_value_count)
            gold_value_counts.append(gold_labels_sel_value_count)

        print(f"Cluster {cluster_id} annotator")
        annotator_value_count_df = pd.concat(annotator_value_counts, axis=1)
        annotator_value_count_df.columns = annotator_names
        annotator_value_count_df = annotator_value_count_df.melt(ignore_index=False, var_name="annotator").reset_index()
        annotator_value_count_df.columns = ["class", "annotator", "value"]
        annotator_value_count_df = annotator_value_count_df.pivot(index="annotator", columns="class")
        annotator_value_count_df.columns = annotator_value_count_df.columns.droplevel(0)
        # Print description of the cluster
        display(annotator_value_count_df.describe())

        print(f"Cluster {cluster_id} gold")
        gold_value_count_df = pd.concat(gold_value_counts, axis=1)
        gold_value_count_df.columns = annotator_names
        gold_value_count_df = gold_value_count_df.melt(ignore_index=False, var_name="annotator").reset_index()
        gold_value_count_df.columns = ["class", "annotator", "value"]
        gold_value_count_df = gold_value_count_df.pivot(index="annotator", columns="class")
        gold_value_count_df.columns = gold_value_count_df.columns.droplevel(0)
        # Print description of the cluster
        display(gold_value_count_df.describe())

        # All texts annotated by annotators in this cluster
        print(f"Clusters {cluster_id} gold aggregated")
        annotator_annotations = data_wide[annotator_names].dropna(how="all")
        gold_labels_selected = gold_labels[gold_labels["id"].isin(annotator_annotations.index)]
        gold_labels_sel_value_count = gold_labels_selected["label"].value_counts(normalize=True)
        display(gold_labels_sel_value_count)

        if plot:
            ax = sns.boxplot(data=annotator_value_count_df)
            ax.set(ylim=(0, 1))
            plt.show(ax)


def plot_cm_cluster_vs_gold_labels(cluster_label_df, gold_label, cluster_labels):
    combinations = [[gold_label, cluster_label] for cluster_label in cluster_labels]
    for combination in combinations:
        selected_df = cluster_label_df[combination]
        total_texts = len(selected_df)
        selected_df = selected_df.dropna()
        cluster_texts = len(selected_df)
        percentage_texts = cluster_texts / total_texts
        cm = confusion_matrix(y_true=selected_df.iloc[:, 0], y_pred=selected_df.iloc[:, 1], normalize='true')
        ax = sns.heatmap(cm, vmin=0, vmax=1, annot=True, cbar=False)
        ax.set_title(f"{total_texts} \\ {cluster_texts} ({percentage_texts})")
        ax.set_xlabel(f'Cluster {combination[1]}')
        ax.set_ylabel('Gold labels')
        plt.show(ax)

def group_annotators_by_cluster(clusters):
    cluster_values = list(set(clusters))
    group_list = []
    for value in cluster_values:
        group_list.append([i for i, x in enumerate(clusters) if x == value])
    return group_list

def get_cluster_stats(data_wide, annotator_names_clusters):
    """
    Returns the number of annotators, number of annotations and number of comments in each cluster
    """
    stats_per_cluster = {}
    for cluster_id, annotator_names in enumerate(annotator_names_clusters):
        selected_df = data_wide[annotator_names]
        selected_df = selected_df.dropna(how="all")
        num_texts = selected_df.shape[0]
        num_annotators = selected_df.shape[1]
        num_annotations = sum(selected_df.count())
        stats_per_cluster[cluster_id] = {"#Annotators": num_annotators, "#Texts": num_texts, "#Annotations": num_annotations}
    return pd.DataFrame(stats_per_cluster).T


def get_annotations_for_cluster(data_wide, annotator_names_clusters):
    """
    Returns the annotations and annotated text for each cluster.
    """
    annotations_per_cluster = []
    for cluster_id, annotator_names in enumerate(annotator_names_clusters):
        columns_to_select = ["text", *annotator_names]
        selected_df = data_wide[columns_to_select]
        selected_df = selected_df.dropna(thresh=2) # Threshold 2 only if text is in the columns
        annotations_per_cluster.append(selected_df)
    return annotations_per_cluster

def get_shared_annotations(annotations_per_cluster):
    """
    Returns the texts that are annotated by each cluster.
    """

    shared_annotations_dict = {}
    shared_annotations = pd.concat(annotations_per_cluster, axis=1, join="inner")
    shared_annotations = shared_annotations.loc[:, ~shared_annotations.columns.duplicated()]
    shared_annotations_dict["full"] = shared_annotations

    for cluster_id, annotations in enumerate(annotations_per_cluster):
        shared_annotations_cluster = shared_annotations.loc[:, annotations.columns]
        shared_annotations_dict[cluster_id] = shared_annotations_cluster
    return shared_annotations_dict

def get_train_test_split_from_dict(cluster_dfs_dict, test_size=0.2, seed=2):
    train_test_cluster_dict = {}
    for cluster_key, cluster_df in cluster_dfs_dict.items():
        cluster_df_train, cluster_df_test = train_test_split(cluster_df, test_size=test_size, random_state=seed)
        train_test_cluster_dict[cluster_key] = {"train": cluster_df_train, "test": cluster_df_test}
    return train_test_cluster_dict
