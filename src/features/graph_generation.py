import pandas as pd
import krippendorff
import numpy as np
import math
from tqdm import tqdm
import networkx as nx
import networkx.algorithms.community as nxcom
import src.features.krippendorffs_alpha as ka
#import community as community_louvain
import collections
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

def getWeightKrippendorffMatrix(df, min_overlap=1, level="nominal"):
    df_matrix = df.to_numpy()
    number_workers = np.size(df_matrix,1)
    # empty distance matrix
    distance_matrix = np.zeros((number_workers,number_workers))
    list_over = []
    for i in tqdm(range(0,number_workers)):
        for j in range(i+1,number_workers):
            weight = 0
            annotator_1 = df_matrix[:, i]
            annotator_2 = df_matrix[:, j]

            annotator_1_cleaned = annotator_1[~np.isnan(annotator_2)]
            annotator_2_cleaned = annotator_2[~np.isnan(annotator_1)]

            annotator_1_cleaned = annotator_1_cleaned[~np.isnan(annotator_1_cleaned)]
            annotator_2_cleaned = annotator_2_cleaned[~np.isnan(annotator_2_cleaned)]

            len_overlap = len(annotator_2_cleaned)
            list_over.append(len_overlap)

            if len_overlap >= min_overlap:
                try:
                    if level=="ordinal":
                        kd_value = krippendorff.alpha(reliability_data=[annotator_1_cleaned,annotator_2_cleaned],
                                          level_of_measurement='ordinal')
                    elif level=="nominal":
                        kd_value = krippendorff.alpha(reliability_data=[annotator_1_cleaned,annotator_2_cleaned],
                                          level_of_measurement='nominal')
                    else:
                        print(f'Level {level} not supported')
                except RuntimeWarning:
                    print(annotator_1_cleaned)
                    print(annotator_2_cleaned)
                if kd_value < 2:
                    weight = 0.5+ (kd_value +1)/(2)
                    #weight = 0.5+ kd_value


            distance_matrix[i,j] = weight
            distance_matrix[j,i] = weight

    return (distance_matrix,list_over)

def create_graph(distance_matrix):
    return nx.from_numpy_array(distance_matrix, create_using=nx.Graph)

# function to normalize data
def changeWeights(G,change):
    for u,v,a in G.edges(data=True):
        a['weight'] = a['weight'] - change
    return G

def plot_graph(G):
    f = nx.draw_networkx(G,
    pos=nx.spring_layout(G, k=0.1),
    node_size=1,
    #edgelist=[],
    edge_color="#000000",
    alpha=0.91,
    with_labels=False)
    plt.show()

def save(G):
    model_path = "../../models/clusters"
    model_id = str(uuid.uuid4())
    model_name = f'{model_id}_graph.gpickle'
    complete_path = os.path.join(pathlib.Path(__file__).parent.absolute(), model_path, model_name)
    nx.write_gpickle(G, complete_path)
    print(f"Saved graph in {complete_path}")
    return model_name

def load(file_name):
    model_path = "../../models/clusters"
    complete_path = os.path.join(pathlib.Path(__file__).parent.absolute(), model_path, file_name)
    G = nx.read_gpickle(complete_path)
    print(f"Loaded {file_name}")
    return G


def getCommunityGroups(G,method="louvian",random_state=0):
    groups = []
    if method == 'louvian':
        partition = community_louvain.best_partition(G, weight="weight",random_state=random_state,resolution=1)
        cnt = collections.Counter()
        for i in partition.values():
            cnt[i] += 1
        for part in cnt:
            groups.append([])
        for key in partition:
            groups[partition[key]].append(key)

    if method == 'greedy':
        groups_frozensets = list(nxcom.greedy_modularity_communities(G, weight="weight"))
        groups = [list(x) for x in groups_frozensets]

    return groups

def addCommunityToGraph(G, group_list):
    cn = 0
    for c in group_list:
        cn = cn + 1
        for v in c:
            # Add 1 to save 0 for external edges
            G.nodes[v]['community'] = cn
    return G

def get_color(i, r_off=1, g_off=1, b_off=1):
    '''Assign a color to a vertex.'''
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)
    return sns.color_palette("hls", 12)[i]


def getStatsOfGroups(group_list):
    output = ""
    output += "Number of communities:\t"+str(len(group_list))+"\n"
    output += "Communities by size:\n"

    sizes = []
    for i in range(0,len(group_list)):
        sizes.append(len(group_list[i]))

    sizes.sort(reverse=True)

    for size in sizes:
        output += "\t" + str(size) + "\n"

    return output

def getExtendedStatsOfGroups(group_list, data_long):
    list_of_lists = []
    for group in group_list:
        selected = data_long.loc[data_long['annotator'].isin(group)]
        list_of_lists.append([len(group), len(selected), len(selected.groupby('id').nunique())])

    return pd.DataFrame(list_of_lists, columns=['#Annotators','#Annotations','#Comments'])


def compute_krippendorffs_alpha_per_cluster(group_list, data_wide):
    if "text" in data_wide.columns:
        data_wide = data_wide.drop(columns=["text"])
    if "id" in data_wide.columns:
        data_wide = data_wide.drop(columns=["id"])

    list_type = type(group_list[0][0])
    cluster_infos = []
    for cluster_id, group in enumerate(group_list):
        if list_type is str:
            selected_annotators = data_wide.loc[:, group]
        elif list_type is int:
            selected_annotators = data_wide.iloc[:, group]
        else:
            raise ValueError("Group_list is of wrong type")
        alpha = ka.apply_krippendorffs_alpha(selected_annotators, "nominal")
        cluster_info = {"Cluster": cluster_id, "Annotator IDs": group, "Annotator Names": selected_annotators.columns.values, "Alpha": alpha}
        cluster_infos.append(cluster_info)
    df = pd.DataFrame(cluster_infos)
    return df

def removeSmallGroup(group_list_input,min_size_group):
    group_list = copy.deepcopy(group_list_input)
    for i in range(len(group_list)-1, -1,-1):
        length = len(group_list[i])
        if length < min_size_group:
            group_list.pop(i)
            #print("Deleted group with index", i,"and size of",length)
    return group_list
