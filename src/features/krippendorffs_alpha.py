import pandas as pd
import numpy as np
import krippendorff
from tqdm import tqdm

def split_by_demographic_for_krippendorff(annotations_df, demographic, dataset, ordinal=False):
    data = {}
    data_df = {}
    data_desc = {}

    unique_dem_instances = annotations_df[demographic].unique()
    print(f"Splitting data in {len(unique_dem_instances)} sets")
    for instance in tqdm(unique_dem_instances):
        selected_df = annotations_df[annotations_df[demographic] == instance]
        if ordinal:
            assert(dataset in ["toxicity", "aggression"])
            selected_df = selected_df[["rev_id", "worker_id", f"{dataset}_score"]]
            selected_pivot = selected_df.pivot(index="worker_id", columns="rev_id", values=f"{dataset}_score")
        else:
            selected_df = selected_df[["rev_id", "worker_id", f"{dataset}"]]
            selected_pivot = selected_df.pivot(index="worker_id", columns="rev_id", values=dataset)
        data_df[instance] = selected_pivot
        data[instance] = selected_pivot.to_numpy()
        num_workers, num_comments = data[instance].shape
        num_annotations = np.count_nonzero(~np.isnan(data[instance]))
        data_desc[instance] = {"num_workers": num_workers, "num_comments": num_comments, "num_annotations": num_annotations}
    data_desc_df = pd.DataFrame.from_dict(data_desc, orient="index")
    return data, data_df, data_desc_df


def apply_krippendorffs_alpha(annotations_wide, level):
    """
    Input in wide from with first column is rev_id and the remaining ones are
    individual annotators.

    """
    if "rev_id" in annotations_wide.columns:
        annotations_wide = annotations_wide.drop(columns=["rev_id"])

    if "text" in  annotations_wide.columns:
        annotations_wide = annotations_wide.drop(columns=["text"])

    if "id" in  annotations_wide.columns:
        annotations_wide = annotations_wide.drop(columns=["id"])

    reliability_data = annotations_wide.T.to_numpy()
    if level== "nominal":
        return krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='nominal')
    elif level=="ordinal":
        return krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='ordinal')
    else:
        print(f'Level {level} not available')


def apply_krippendorffs_alpha_per_group(annotations_wide_dict, level):
    alphas = {}
    for key, annotations in tqdm(annotations_wide_dict.items()):
            alphas[key] = apply_krippendorffs_alpha(annotations, level)
    return alphas
