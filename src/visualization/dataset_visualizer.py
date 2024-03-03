import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_classes(dataset, annotations_dem_df, level="nominal"):
    pass

def plot_multiple_demographics(df_dict):
    plots = []
    for group, df in df_dict.items():
        plot = plot_demographics(df, group)
        plots.append(plot)
    return plots

def plot_demographics(df, title_desc):
        gender_count = df.gender.value_counts()
        english_count = df.english_first_language.value_counts()
        age_group_count = df.age_group.value_counts()
        education_count = df.education.value_counts()

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, sharey=True)
        gender_count.plot(kind="bar", ax=ax1, title="Gender")
        english_count.plot(kind="bar", ax=ax2, title= "English native")
        age_group_count.plot(kind="bar", ax=ax3, title = "Age group")
        education_count.plot(kind="bar", ax=ax4, title = "Education")
        fig.suptitle("Demographics of {} annotators".format(title_desc))
        return fig

#TODO: DawidSkeneVisualizer
# TSNE, PCA,
# Input 1: df with columns: reliability, positivism, pessimism
# Input 2: dict with dfs as above
