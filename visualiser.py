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

        sns.lineplot(ax=ax[0, 0],
                     data=df,
                     x="iter",
                     y="num.topics")
        ax[0, 0].title.set_text("(a)")

        sns.lineplot(ax=ax[0, 1],
                     data=df,
                     x="iter",
                     y="num.tables")
        ax[0, 1].title.set_text("(b)")

        sns.distplot(ax=ax[1, 0],
                     a=df["num.topics"],
                     bins=df["num.topics"].nunique(),
                     kde=True,
                     kde_kws={"bw": 1})
        ax[1, 0].title.set_text("(c)")

        sns.distplot(ax=ax[1, 1],
                     a=df["num.tables"],
                     bins=df["num.tables"].nunique(),
                     kde=True,
                     kde_kws={"bw": 1})
        ax[1, 1].title.set_text("(d)")

        plt.tight_layout()


if __name__ == "__main__":
    vis = Visualiser(out_dir="./__output__/vis")
    s = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Datasets\pan17_train"
         r"\problem001\hdp_lss_common_0.5_1_1\state.log")
    vis.plot_gibbs_trace(data=s)
