# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:59:09 2019
A class which caters to visualisation

@author: RTRAD
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Visualiser():
    # Predefined styles values
    style_whitegrid = "whitegrid"
    style_darkgrid = "darkgrid"
    style_dark = "dark"
    style_white = "white"
    style_ticks = "ticks"

    def __init__(self,
                 out_dir=str,
                 style: str = "darkgrid",
                 dpi: int = 600,
                 charts_format: str = "pdf"):
        # Set seaborn defaults
        sns.set()
        sns.set_style(style)
        self.output_dir = out_dir
        self.dpi = dpi
        self.format = charts_format

    def plot_gibbs_trace(self, data: str):
        df = pd.read_csv(
                filepath_or_buffer=data,
                delim_whitespace=True,
                usecols=["iter", "time", "likelihood",
                         "num.tables",  "num.topics"])
        fig, ax = plt.subplots(nrows=2,
                               ncols=2,
                               clear=True,
                               figsize=(9, 9))
        for a in ax:
            for b in a:
                b.margins(x=0)

        ax[0, 0].set_title("(a)")
        sns.lineplot(ax=ax[0, 0],
                     data=df,
                     x="iter",
                     y="num.topics")

        ax[0, 1].set_title("(b)")
        sns.lineplot(ax=ax[0, 1],
                     data=df,
                     x="iter",
                     y="num.tables")

        ax[1, 0].set_title("(c)")
        sns.distplot(ax=ax[1, 0],
                     a=df["num.topics"],
                     bins=df["num.topics"].nunique(),
                     kde=True,
                     kde_kws={"bw": 1})

        ax[1, 1].set_title("(d)")
        sns.distplot(ax=ax[1, 1],
                     a=df["num.tables"],
                     bins=df["num.tables"].nunique(),
                     kde=True,
                     kde_kws={"bw": 1})

        plt.tight_layout()

    def analyse_results(self,
                        save_dir: str,
                        sparse_path: str,
                        neutral_path: str,
                        dense_path: str):

        df_neutral_res = pd.read_csv(neutral_path, low_memory=False)
        df_neutral_res = df_neutral_res[df_neutral_res.algorithm != "TRUE"]
        df_sparse_res = pd.read_csv(sparse_path, low_memory=False)
        df_sparse_res = df_sparse_res[df_sparse_res.algorithm != "TRUE"]
        df_dense_res = pd.read_csv(dense_path, low_memory=False)
        df_dense_res = df_dense_res[df_dense_res.algorithm != "TRUE"]

        df_all = pd.concat(
                [df_neutral_res, df_sparse_res, df_dense_res],
                axis=0,
                keys=["Neutral", "Sparse", "Dense"]
                ).reset_index(level=0).rename(
                        columns={"level_0": "topics_prior"})

        fig_overall, ax_overall = plt.subplots(nrows=2,
                                               ncols=1,
                                               clear=True,
                                               figsize=(9, 15))
        sns.barplot(x="algorithm", y="bcubed_fscore", hue="topics_prior",
                    data=df_all,
                    capsize=0.05,
                    ax=ax_overall[0])
        sns.barplot(x="algorithm", y="ari", hue="topics_prior",
                    data=df_all,
                    capsize=0.05,
                    ax=ax_overall[1])

        # Rotate the x labels:
        for ax in fig_overall.axes:
            plt.sca(ax)
            ax.set_xlabel("")
            plt.xticks(rotation=90)
        plt.tight_layout()

        return df_all


if __name__ == "__main__":
    vis = Visualiser(out_dir="./__output__/vis")
    s = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Datasets\pan17_train"
         r"\problem001\hdp_lss_0.50_1.00_1.00_common_True\state.log")
#    vis.plot_gibbs_trace(data=s)

    neutral_path = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
                    r"\authorship_clustering_code_repo\__outputs__"
                    r"\results_20190820_213150_training_neutral_common.csv")

    sparse_path = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
                   r"\authorship_clustering_code_repo\__outputs__"
                   r"\results_20190821_132024_training_sparse_common.csv")

    dense_path = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
                  r"\authorship_clustering_code_repo\__outputs__"
                  r"\results_20190820_230536_training_dense_common.csv")

    df = vis.analyse_results(save_dir="",
                             sparse_path=sparse_path,
                             neutral_path=neutral_path,
                             dense_path=dense_path)
