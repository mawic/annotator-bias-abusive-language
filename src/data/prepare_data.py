import krippendorff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.data.data_loader import (
    EastAsianPrejudiceDataset,
    MisogynyDataset,
    RedditDataset,
    WikipediaDataset,
)
from src.features.annotation_aggregator import (
    DawidSkeneAggregator,
    MaceAggregator,
    MajorityVoteAggregator,
)
from tqdm.notebook import tqdm


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

        try:
            reliability_data = [relevant_gold_labels, labels_workers_annotations]
            ira_scores.append(
                krippendorff.alpha(
                    reliability_data=reliability_data, level_of_measurement="nominal"
                )
            )
        except Warning:
            print(reliability_data)
    ira_scores = np.asarray(ira_scores)
    return worker_ids, ira_scores


def get_annotator_and_gold_labels_EastAsian():
    dataset = EastAsianPrejudiceDataset()
    expert_data = dataset.get_data(expert_only=True, binary=True)

    mapping = { "neutral": 0.0, "hostility": 1.0}

    annotator_data_wide = dataset.get_data(
        expert_only=False, binary=True, format="wide"
    )
    annotator_data_wide = annotator_data_wide.sort_values("id").reset_index()
    for col in annotator_data_wide.columns[2:]:
        annotator_data_wide = annotator_data_wide.replace({col: mapping})

    expert_data = expert_data.sort_values("id").reset_index()
    expert_data = expert_data.replace({"label": mapping})

    expert_data = expert_data[
        expert_data["id"].isin(annotator_data_wide["id"].to_list())
    ]
    annotator_data_wide = annotator_data_wide[
        annotator_data_wide["id"].isin(expert_data["id"].to_list())
    ]

    annotator_data_long = dataset.get_data(
        expert_only=False, binary=True, format="long"
    )
    annotator_data_long = annotator_data_long.replace({'label': mapping})
    annotator_data_long = annotator_data_long.reset_index()

    return mapping, expert_data, annotator_data_wide,annotator_data_long


def get_annotator_and_gold_labels_Reddit():
    dataset = RedditDataset()
    expert_data = dataset.get_data(expert_only=True, binary=True)

    mapping = {"non-derogatory": 0.0, "derogatory": 1.0}

    annotator_data_wide = dataset.get_data(
        expert_only=False, binary=True, format="wide"
    )
    # annotator_data_wide = annotator_data_wide.sort_values("id").reset_index()
    for col in annotator_data_wide.columns[2:]:
        annotator_data_wide = annotator_data_wide.replace({col: mapping})

    expert_data = expert_data.sort_values("id")
    expert_data = expert_data.replace({"label": mapping})
    expert_data = expert_data[
        expert_data["id"].isin(annotator_data_wide["id"].to_list())
    ]
    annotator_data_wide = annotator_data_wide[
        annotator_data_wide["id"].isin(expert_data["id"].to_list())
    ]
    
    annotator_data_long = dataset.get_data(
        expert_only=False, binary=True, format="long"
    )
    annotator_data_long = annotator_data_long.replace({'label': mapping})

    return mapping, expert_data, annotator_data_wide,annotator_data_long


def get_annotator_and_gold_labels_Wikipedia():
    dataset = "attack"
    wiki_data = WikipediaDataset(dataset)

    # Getting gold labels first by loading all data and aggregating by majority vote
    level = "nominal"

    all_annotations_long = wiki_data.get_annotations(
        kind="df", return_demographics=False, level=level
    )
    all_annotations_long = all_annotations_long.rename(
        columns={
            "rev_id": "id",
            "comment": "text",
            "worker_id": "annotator",
            "annotation": "label",
        }
    )

    mj = MajorityVoteAggregator()
    mj.aggregate(all_annotations_long)
    mj_gold_labels = mj.get_aggregation(kind="df", return_text=False)
    mj_gold_labels = mj_gold_labels.reset_index()

    mj_gold_labels = mj_gold_labels.rename(columns={"index": "id"})

    all_annotations_wide = wiki_data.get_annotations(
        kind="matrix", return_demographics=False, level=level
    )
    all_annotations_wide = all_annotations_wide.rename(
        columns={"rev_id": "id", "comment": "text"}
    )
    mapping = {"Non-attack": 0.0, "Attack": 1.0}

    return mapping, mj_gold_labels, all_annotations_wide, all_annotations_long


def get_annotator_and_gold_labels_Misogyny():
    dataset = MisogynyDataset()
    expert_data = dataset.get_data(expert_only=True, binary=True)

    mapping = {"Nonmisogynistic": 0.0, "Misogynistic": 1.0}

    annotator_data_wide = dataset.get_data(
        expert_only=False, binary=True, format="wide"
    )
    # annotator_data_wide = annotator_data_wide.sort_values("id").reset_index()
    for col in annotator_data_wide.columns[2:]:
        annotator_data_wide = annotator_data_wide.replace({col: mapping})

    expert_data = expert_data.sort_values("id")
    expert_data = expert_data.replace({"label": mapping})
    expert_data = expert_data[
        expert_data["id"].isin(annotator_data_wide["id"].to_list())
    ]
    annotator_data_wide = annotator_data_wide[
        annotator_data_wide["id"].isin(expert_data["id"].to_list())
    ]

    annotator_data_long = dataset.get_data(
        expert_only=False, binary=True, format="long"
    )
    annotator_data_long = annotator_data_long.replace({'label': mapping})

    return mapping, expert_data, annotator_data_wide,annotator_data_long


