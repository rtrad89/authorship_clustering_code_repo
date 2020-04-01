# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:59:09 2019
A class which caters to visualisation

@author: RTRAD
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from aiders import Tools
from matplotlib.colors import ListedColormap


class Visualiser():
    # Predefined styles values
    style_whitegrid = "whitegrid"
    style_darkgrid = "darkgrid"
    style_dark = "dark"
    style_white = "white"
    style_ticks = "ticks"

    def __init__(self,
                 rc: dict,
                 style: str = "darkgrid",
                 error_width: float = 1.0,
                 errorcap_size: float = 0.05,
                 single_size: tuple = (5, 5),
                 square_size: tuple = (9, 9),
                 portrait_size: tuple = (9, 15),
                 landscape_size: tuple = (15, 9),
                 rectangle_size: tuple = (15, 5),
                 double_square_size: tuple = (15, 7.5),
                 large_square_size: tuple = (21, 21)):
        # Set seaborn defaults
        sns.set(rc=rc)
        sns.set_style(style)
        self.error_width = error_width
        self.error_cap_size = errorcap_size
        self.single = single_size
        self.square = square_size
        self.portrait = portrait_size
        self.landscape = landscape_size
        self.rectangle = rectangle_size
        self.double_square = double_square_size
        self.large_square = large_square_size
        self.figs = {}

    def show_values_on_bars(self, axs, size: int = 10):
        def _show_on_single_plot(ax):
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = .5 * (p.get_y() + p.get_height())
                value = f"{p.get_height():0.3f}"
                ax.text(_x, _y, value, ha="center",
                        fontdict={"color": "black",
                                  "weight": "bold",
                                  "size": size})

        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                _show_on_single_plot(ax)
        else:
            _show_on_single_plot(axs)

    def plot_gibbs_trace(self,
                         state_path: str,
                         key_suff: str):
        df = pd.read_csv(
                filepath_or_buffer=state_path,
                delim_whitespace=True,
                usecols=["iter", "time", "likelihood",
                         "num.tables",  "num.topics"])
        fig, ax = plt.subplots(nrows=2,
                               ncols=2,
                               clear=True,
                               figsize=self.square)
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
        self.figs.update({f"Gibbs_Trace{key_suff}": fig})
        plt.tight_layout()
        plt.show()
        plt.close()

    def plot_elaborate_gibbs_trace(self,
                                   state_path: str,
                                   key_suff: str):
        df = pd.read_csv(
                filepath_or_buffer=state_path,
                delim_whitespace=True,
                usecols=["iter", "time", "likelihood",
                         "num.tables",  "num.topics"])
        fig1, ax1 = plt.subplots(nrows=2,
                                 ncols=1,
                                 clear=True,
                                 figsize=self.square)

        fig2, ax2 = plt.subplots(nrows=2,
                                 ncols=1,
                                 clear=True,
                                 figsize=self.square)

        for a in ax1:
            a.margins(x=0)

        for a in ax2:
            a.margins(x=0)

        ax1[0].set_title("(a)")
        sns.lineplot(ax=ax1[0],
                     data=df,
                     x="iter",
                     y="num.topics")

        ax1[1].set_title("(b)")
        sns.lineplot(ax=ax1[1],
                     data=df,
                     x="iter",
                     y="num.tables")

        ax2[0].set_title("(c)")
        sns.distplot(ax=ax2[0],
                     a=df["num.topics"],
                     bins=df["num.topics"].nunique(),
                     kde=True,
                     kde_kws={"bw": 1})

        ax2[1].set_title("(d)")
        sns.distplot(ax=ax2[1],
                     a=df["num.tables"],
                     bins=df["num.tables"].nunique(),
                     kde=True,
                     kde_kws={"bw": 1})
        self.figs.update({f"Gibbs_Trace1_{key_suff}": fig1,
                          f"Gibbs_Trace2_{key_suff}": fig2})
        plt.tight_layout()
        plt.show()
        plt.close()

    def analyse_results_compact(self,
                                concise: bool,
                                test_style: bool,
                                sparse_path: str,
                                neutral_path: str,
                                dense_path: str,
                                key_suff: str):

        df_neutral_res = pd.read_csv(neutral_path, low_memory=False)
        df_neutral_res = df_neutral_res[df_neutral_res.algorithm != "Labels"]
        df_sparse_res = pd.read_csv(sparse_path, low_memory=False)
        df_sparse_res = df_sparse_res[df_sparse_res.algorithm != "Labels"]
        df_dense_res = pd.read_csv(dense_path, low_memory=False)
        df_dense_res = df_dense_res[df_dense_res.algorithm != "Labels"]

        # Filter out redundant algorithms:
        if concise:
            df_neutral_res = df_neutral_res[
                    ~df_neutral_res.algorithm.isin(
                            ["E_HDBSCAN", "E_HAC_Single", "E_HAC_Average"])]
            df_sparse_res = df_sparse_res[
                    ~df_sparse_res.algorithm.isin(
                            ["E_HDBSCAN", "E_HAC_Single", "E_HAC_Average"])]
            df_dense_res = df_dense_res[
                    ~df_dense_res.algorithm.isin(
                            ["E_HDBSCAN", "E_HAC_Single", "E_HAC_Average"])]

        if test_style:
            df_all = df_sparse_res
        else:
            df_all = pd.concat(
                    [df_neutral_res, df_sparse_res, df_dense_res],
                    axis=0,
                    keys=["Neutral", "Sparse", "Dense"]
                    ).reset_index(level=0).rename(
                            columns={"level_0": "topics_prior"})

        algo_order = ["BL_r", "BL_s", "BL_SOTA_tf",
                      "BL_SOTA_tfidf", "BL_SOTA_le",
                      "S_HDP",
                      "E_HAC_C", "E_Mean_Shift", "E_OPTICS", "E_SPKMeans"]

        fig_overall, ax_overall = plt.subplots(nrows=1,
                                               ncols=2,
                                               clear=True,
                                               figsize=self.rectangle)
        if test_style:
            sns.barplot(x="algorithm", y="bcubed_fscore",
                        data=df_all,
                        # Use the simpler standard deviation instead of CI
                        ci="sd",
                        errwidth=self.error_width,
                        capsize=self.error_cap_size,
                        order=algo_order,
                        ax=ax_overall[0])

            sns.barplot(x="algorithm", y="ari",
                        data=df_all,
                        # Use the simpler standard deviation instead of CI
                        ci="sd",
                        errwidth=self.error_width,
                        capsize=self.error_cap_size,
                        order=algo_order,
                        ax=ax_overall[1])
        else:
            sns.barplot(x="algorithm", y="bcubed_fscore", hue="topics_prior",
                        data=df_all,
                        # Use the simpler standard deviation instead of CI
                        ci="sd",
                        errwidth=self.error_width,
                        capsize=self.error_cap_size,
                        order=algo_order,
                        ax=ax_overall[0])

            sns.barplot(x="algorithm", y="ari", hue="topics_prior",
                        data=df_all,
                        # Use the simpler standard deviation instead of CI
                        ci="sd",
                        errwidth=self.error_width,
                        capsize=self.error_cap_size,
                        order=algo_order,
                        ax=ax_overall[1])

        if not test_style:
            ax_overall[0].legend(loc="lower right")
            ax_overall[1].legend(loc="upper left")
        else:
            # Set the state-of-the-art bar:
            ax_overall[0].axhline(0.573, ls='--')

        # Rotate the x labels:
        for ax in fig_overall.axes:
            plt.sca(ax)
            ax.set_xlabel("")
            plt.xticks(rotation=90)
        # Show values on final test charts:
        if test_style:
            self.show_values_on_bars(ax_overall)
        plt.tight_layout()
        plt.show()
        plt.close()

        fig_genre_lang, ax_genre_lang = plt.subplots(nrows=2,
                                                     ncols=2,
                                                     clear=True,
                                                     figsize=self.landscape,
                                                     sharex="col",
                                                     sharey="row")
        sns.barplot(x="algorithm", y="bcubed_fscore", hue="language",
                    data=df_sparse_res,
                    # Use the simpler standard deviation instead of CI
                    ci="sd",
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    order=algo_order,
                    ax=ax_genre_lang[0, 0])
        sns.barplot(x="algorithm", y="bcubed_fscore", hue="genre",
                    data=df_sparse_res,
                    # Use the simpler standard deviation instead of CI
                    ci="sd",
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    order=algo_order,
                    ax=ax_genre_lang[0, 1])
        sns.barplot(x="algorithm", y="ari", hue="language",
                    data=df_sparse_res,
                    # Use the simpler standard deviation instead of CI
                    ci="sd",
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    order=algo_order,
                    ax=ax_genre_lang[1, 0])
        sns.barplot(x="algorithm", y="ari", hue="genre",
                    data=df_sparse_res,
                    # Use the simpler standard deviation instead of CI
                    ci="sd",
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    order=algo_order,
                    ax=ax_genre_lang[1, 1])

        # Rotate the x axes
        for ax in fig_genre_lang.axes:
            plt.sca(ax)
            plt.xticks(rotation=90)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
        plt.close()

        fig_comb_genre_lang, ax_comb_genre_lang = plt.subplots(
                nrows=2,
                ncols=3,
                clear=True,
                figsize=self.landscape,
                sharex="col",
                sharey="row")

        sns.barplot(x="algorithm", y="bcubed_fscore", hue="genre",
                    data=df_sparse_res[df_sparse_res.language == "en"],
                    # Use the simpler standard deviation instead of CI
                    ci="sd",
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    order=algo_order,
                    ax=ax_comb_genre_lang[0, 0])
        ax_comb_genre_lang[0, 0].set_title("English")
        sns.barplot(x="algorithm", y="bcubed_fscore", hue="genre",
                    data=df_sparse_res[df_sparse_res.language == "nl"],
                    # Use the simpler standard deviation instead of CI
                    ci="sd",
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    ax=ax_comb_genre_lang[0, 1])
        ax_comb_genre_lang[0, 1].set_title("Dutch")
        sns.barplot(x="algorithm", y="bcubed_fscore", hue="genre",
                    data=df_sparse_res[df_sparse_res.language == "gr"],
                    # Use the simpler standard deviation instead of CI
                    ci="sd",
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    order=algo_order,
                    ax=ax_comb_genre_lang[0, 2])
        ax_comb_genre_lang[0, 2].set_title("Greek")

        sns.barplot(x="algorithm", y="ari", hue="genre",
                    data=df_sparse_res[df_sparse_res.language == "en"],
                    # Use the simpler standard deviation instead of CI
                    ci="sd",
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    order=algo_order,
                    ax=ax_comb_genre_lang[1, 0])
        sns.barplot(x="algorithm", y="ari", hue="genre",
                    data=df_sparse_res[df_sparse_res.language == "nl"],
                    # Use the simpler standard deviation instead of CI
                    ci="sd",
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    order=algo_order,
                    ax=ax_comb_genre_lang[1, 1])
        sns.barplot(x="algorithm", y="ari", hue="genre",
                    data=df_sparse_res[df_sparse_res.language == "gr"],
                    # Use the simpler standard deviation instead of CI
                    ci="sd",
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    order=algo_order,
                    ax=ax_comb_genre_lang[1, 2])

        for ax in fig_comb_genre_lang.axes:
            plt.sca(ax)
            ax.set_xlabel("")
            plt.xticks(rotation=90)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
        plt.close()

        self.figs.update(
                {f"Results_Overall{key_suff}": fig_overall,
                 f"Results_genre_lang{key_suff}": fig_genre_lang,
                 f"Results_combined_genre_lang{key_suff}": fig_comb_genre_lang
                 })

        return df_all

    def analyse_results(self,
                        concise: bool,
                        test_style: bool,
                        sparse_path: str,
                        key_suff: str,
                        neutral_path: str = None,
                        dense_path: str = None):
        # TODO: segment the visualisation to reduce if statements count
        df_sparse_res = pd.read_csv(sparse_path, low_memory=False,
                                    usecols=["set", "algorithm", "ari",
                                             "bcubed_fscore", "language",
                                             "genre"])
        df_sparse_res = df_sparse_res[df_sparse_res.algorithm != "Labels"]

        if neutral_path:
            df_neutral_res = pd.read_csv(neutral_path, low_memory=False,
                                         usecols=["set", "algorithm", "ari",
                                                  "bcubed_fscore", "language",
                                                  "genre"])
            df_neutral_res = df_neutral_res[
                df_neutral_res.algorithm != "Labels"]

        if dense_path:
            df_dense_res = pd.read_csv(dense_path, low_memory=False,
                                       usecols=["set", "algorithm", "ari",
                                                "bcubed_fscore", "language",
                                                "genre"])
            df_dense_res = df_dense_res[df_dense_res.algorithm != "Labels"]

        # Filter out redundant algorithms:
        if concise:
            df_sparse_res = df_sparse_res[
                    ~df_sparse_res.algorithm.isin(
                            ["E_HDBSCAN", "E_HAC_Single", "E_HAC_Average"])]
            if neutral_path:
                df_neutral_res = df_neutral_res[
                        ~df_neutral_res.algorithm.isin(
                                ["E_HDBSCAN", "E_HAC_Single", "E_HAC_Average"]
                                )]
            if dense_path:
                df_dense_res = df_dense_res[
                        ~df_dense_res.algorithm.isin(
                                ["E_HDBSCAN", "E_HAC_Single", "E_HAC_Average"]
                                )]

        if test_style:
            df_all = df_sparse_res
        elif neutral_path and dense_path:
            df_all = pd.concat(
                    [df_neutral_res, df_sparse_res, df_dense_res],
                    axis=0,
                    keys=["Neutral", "Sparse", "Dense"]
                    ).reset_index(level=0).rename(
                            columns={"level_0": "topics_prior"})
        else:
            print("Neutral and dense results are expected but not provided!")

        algo_order = ["BL_r", "BL_s", "BL_SOTA_tf",
                      "BL_SOTA_tfidf", "BL_SOTA_le",
                      "S_HDP",
                      "E_HAC_C", "E_Mean_Shift", "E_OPTICS", "E_SPKMeans",
                      "E_COP_KMeans"]

        colours = ["#fafafc", "#fafafc",
                   "#84d674", "#84d674", "#84d674",
                   "#f3f571",
                   "#3c6cf0", "#3c6cf0", "#3c6cf0", "#3c6cf0",
                   "#3c6cf0"]
        fig_overall_b3f, ax_overall_b3f = plt.subplots(nrows=1,
                                                       ncols=1,
                                                       clear=True,
                                                       figsize=self.landscape)

        fig_overall_ari, ax_overall_ari = plt.subplots(nrows=1,
                                                       ncols=1,
                                                       clear=True,
                                                       figsize=self.landscape)
        if test_style:
            sns.barplot(x="algorithm", y="bcubed_fscore",
                        data=df_all,
                        # Use the simpler standard deviation instead of CI
                        ci="sd",
                        errwidth=self.error_width,
                        capsize=self.error_cap_size,
                        order=algo_order,
                        ax=ax_overall_b3f,
                        palette=colours,
                        edgecolor=".2")

            sns.barplot(x="algorithm", y="ari",
                        data=df_all,
                        # Use the simpler standard deviation instead of CI
                        ci="sd",
                        errwidth=self.error_width,
                        capsize=self.error_cap_size,
                        order=algo_order,
                        ax=ax_overall_ari,
                        palette=colours,
                        edgecolor=".2")
        else:
            sns.barplot(x="algorithm", y="bcubed_fscore", hue="topics_prior",
                        data=df_all,
                        # Use the simpler standard deviation instead of CI
                        ci="sd",
                        errwidth=self.error_width,
                        capsize=self.error_cap_size,
                        order=algo_order,
                        ax=ax_overall_b3f,
                        palette=colours,
                        edgecolor=".2")

            sns.barplot(x="algorithm", y="ari", hue="topics_prior",
                        data=df_all,
                        # Use the simpler standard deviation instead of CI
                        ci="sd",
                        errwidth=self.error_width,
                        capsize=self.error_cap_size,
                        order=algo_order,
                        ax=ax_overall_ari,
                        palette=colours,
                        edgecolor=".2")

        if not test_style:
            ax_overall_b3f.legend(loc="lower right")
            ax_overall_ari.legend(loc="upper left")
        else:
            # Set the state-of-the-art bar:
            ax_overall_b3f.axhline(0.573, ls='--')

        # Rotate the x labels:
        for ax1 in fig_overall_b3f.axes:
            plt.sca(ax1)
            ax1.set_xlabel("")
            plt.xticks(rotation=90)
        for ax2 in fig_overall_ari.axes:
            plt.sca(ax2)
            ax2.set_xlabel("")
            plt.xticks(rotation=90)

        # Show values on final test charts:
        if test_style:
            self.show_values_on_bars(ax_overall_b3f, size=22)
            self.show_values_on_bars(ax_overall_ari, size=22)
        plt.tight_layout()
        plt.show()
        plt.close()

        fig_comb_genre_lang_en, ax_comb_genre_lang_en = plt.subplots(
                nrows=2,
                ncols=1,
                clear=True,
                figsize=self.portrait,
                sharex="col")

        fig_comb_genre_lang_nl, ax_comb_genre_lang_nl = plt.subplots(
                nrows=2,
                ncols=1,
                clear=True,
                figsize=self.portrait,
                sharex="col")

        fig_comb_genre_lang_gr, ax_comb_genre_lang_gr = plt.subplots(
                nrows=2,
                ncols=1,
                clear=True,
                figsize=self.portrait,
                sharex="col")

        sns.barplot(x="algorithm", y="bcubed_fscore", hue="genre",
                    data=df_sparse_res[df_sparse_res.language == "en"],
                    # Use the simpler standard deviation instead of CI
                    ci="sd",
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    order=algo_order,
                    ax=ax_comb_genre_lang_en[0])
        ax_comb_genre_lang_en[0].set_title("English")
        sns.barplot(x="algorithm", y="bcubed_fscore", hue="genre",
                    data=df_sparse_res[df_sparse_res.language == "nl"],
                    # Use the simpler standard deviation instead of CI
                    ci="sd",
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    order=algo_order,
                    ax=ax_comb_genre_lang_nl[0])
        ax_comb_genre_lang_nl[0].set_title("Dutch")
        sns.barplot(x="algorithm", y="bcubed_fscore", hue="genre",
                    data=df_sparse_res[df_sparse_res.language == "gr"],
                    # Use the simpler standard deviation instead of CI
                    ci="sd",
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    order=algo_order,
                    ax=ax_comb_genre_lang_gr[0])
        ax_comb_genre_lang_gr[0].set_title("Greek")

        sns.barplot(x="algorithm", y="ari", hue="genre",
                    data=df_sparse_res[df_sparse_res.language == "en"],
                    # Use the simpler standard deviation instead of CI
                    ci="sd",
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    order=algo_order,
                    ax=ax_comb_genre_lang_en[1])
        sns.barplot(x="algorithm", y="ari", hue="genre",
                    data=df_sparse_res[df_sparse_res.language == "nl"],
                    # Use the simpler standard deviation instead of CI
                    ci="sd",
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    order=algo_order,
                    ax=ax_comb_genre_lang_nl[1])
        sns.barplot(x="algorithm", y="ari", hue="genre",
                    data=df_sparse_res[df_sparse_res.language == "gr"],
                    # Use the simpler standard deviation instead of CI
                    ci="sd",
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    order=algo_order,
                    ax=ax_comb_genre_lang_gr[1])

        for ax1 in fig_comb_genre_lang_en.axes:
            plt.sca(ax1)
            ax1.set_xlabel("")
            plt.xticks(rotation=90)
        for ax2 in fig_comb_genre_lang_nl.axes:
            plt.sca(ax2)
            ax2.set_xlabel("")
            plt.xticks(rotation=90)
        for ax3 in fig_comb_genre_lang_gr.axes:
            plt.sca(ax3)
            ax3.set_xlabel("")
            plt.xticks(rotation=90)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
        plt.close()

        self.figs.update(
                {f"Results_Overall_ari_{key_suff}": fig_overall_ari,
                 f"Results_Overall_b3f_{key_suff}": fig_overall_b3f,
                 f"Results_combined_genre_lang_en_{key_suff}":
                     fig_comb_genre_lang_en,
                 f"Results_combined_genre_lang_nl_{key_suff}":
                     fig_comb_genre_lang_nl,
                 f"Results_combined_genre_lang_gr_{key_suff}":
                     fig_comb_genre_lang_gr})

        return df_all

    def analyse_true_k_results(self,
                               true_path: str,
                               est_path: str,
                               key_suff: str):
        # Read the data
        df_est = pd.read_csv(est_path,
                             usecols=["set", "algorithm", "ari",
                                      "bcubed_fscore"],
                             low_memory=False)
        df_est = df_est[df_est.algorithm != "Labels"]
        df_est = df_est[
                df_est.algorithm.isin(
                        ["E_HAC_C", "E_SPKMeans", "E_COP_KMeans"])]

        df_true = pd.read_csv(true_path,
                              usecols=["set", "algorithm", "ari",
                                       "bcubed_fscore"],
                              low_memory=False)
        df_true = df_true[df_true.algorithm != "Labels"]
        df_true = df_true[
                df_true.algorithm.isin(
                        ["E_HAC_C", "E_SPKMeans", "E_COP_KMeans"])]

        # Merge the two dataframes
        df_all = df_est.merge(df_true, on=["set", "algorithm"],
                              suffixes=("_est", "_true"))

        # Group the results and plot them
        df_all = df_all.groupby(by=["algorithm"]).mean()
        df_all = df_all.sort_index(axis=1).T
        df_all = df_all.reset_index(
                ).rename(columns={"index": "measurement"}
                         ).melt(id_vars="measurement")

        fig, ax = plt.subplots(nrows=1,
                               ncols=1,
                               clear=True,
                               figsize=self.square)

        sns.barplot(data=df_all, x="algorithm", y="value", hue="measurement",
                    ax=ax)
        # Rotate the axes 90 degrees
        for a in fig.axes:
            plt.sca(a)
            a.set_xlabel("")
            plt.xticks(rotation=90)
        # Annotate the bars
        self.show_values_on_bars(ax, size=13)
        # Output the chart
        plt.legend(loc="upper left", bbox_to_anchor=(0, 0.8))
        plt.tight_layout()
        plt.show()
        plt.close()

        # Save the figure in the cache for later serialisation
        self.figs.update(
                {f"Results_true_k_improvement": fig})
        return df_all

    def analyse_k_trends(self,
                         concise: bool,
                         k_vals_path: str,
                         key_suff: str):
        if concise:
            k_vals = pd.read_csv(k_vals_path, low_memory=False,
                                 usecols=["Gap", "G-means", "E_SPKMeans",
                                          "E_HAC_C", "E_OPTICS",
                                          "E_COP_KMeans",
                                          "TRUE"]
                                 )[["Gap", "G-means", "E_SPKMeans",
                                    "E_HAC_C", "E_OPTICS", "E_COP_KMeans",
                                    "TRUE"]]
        else:
            k_vals = pd.read_csv(k_vals_path, low_memory=False, index_col=0)

        fig, ax = plt.subplots(nrows=2,
                               ncols=2,
                               clear=True,
                               figsize=self.landscape,
                               sharex="col")

        sns.scatterplot(x="TRUE", y="E_SPKMeans", color=".0",
                        marker="X",
                        ax=ax[0, 0],
                        data=k_vals)
        sns.scatterplot(x="TRUE", y="E_HAC_C", color=".0",
                        marker="P",
                        ax=ax[0, 1],
                        data=k_vals)
        sns.scatterplot(x="TRUE", y="E_OPTICS", color=".0",
                        marker="^",
                        ax=ax[1, 0],
                        data=k_vals)
        sns.scatterplot(x="TRUE", y="E_COP_KMeans", color=".0",
                        marker="s",
                        ax=ax[1, 1],
                        data=k_vals)

        if not concise:
            sns.scatterplot(x="TRUE", y="E_HAC_S", color=".0",
                            marker="s",
                            ax=ax[1, 1],
                            data=k_vals)
            sns.scatterplot(x="TRUE", y="E_HAC_A", color=".0",
                            marker="o",
                            ax=ax[1, 2],
                            data=k_vals)

        plt.tight_layout()
        plt.show()
        plt.close()

        # Plot the RMSE trends
        rmse = {}
        for col in k_vals.columns:
            if col in ["TRUE"]:
                continue
            rmse.update({col: Tools.calc_rmse(k_vals["TRUE"], k_vals[col])})

        # Add SOTA estimations
        sota_pred_le = Tools.get_sota_est_k(
                output_path=(r"D:\College\DKEM\Thesis\AuthorshipClustering"
                             r"\Code\clusterPAN2017-master\output_LogEnt"))
        rmse.update(
                {"BL_SOTA_le": Tools.calc_rmse(k_vals["TRUE"], sota_pred_le)})

        sota_pred_tf = Tools.get_sota_est_k(
                output_path=(r"D:\College\DKEM\Thesis\AuthorshipClustering"
                             r"\Code\clusterPAN2017-master\output_Tf"))
        rmse.update(
                {"BL_SOTA_tf": Tools.calc_rmse(k_vals["TRUE"], sota_pred_tf)})

        sota_pred_tfidf = Tools.get_sota_est_k(
                output_path=(r"D:\College\DKEM\Thesis\AuthorshipClustering"
                             r"\Code\clusterPAN2017-master\output_TfIdf"))
        rmse.update(
                {"BL_SOTA_tfidf": Tools.calc_rmse(
                        k_vals["TRUE"], sota_pred_tfidf)})

        df_rmse = pd.DataFrame(data=rmse, index=[0]).T
        df_rmse.columns = ["RMSE"]

        fig_rmse, ax_rmse = plt.subplots(clear=True, figsize=self.square)
        sns.barplot(x=df_rmse.index, y="RMSE", data=df_rmse, ci="sd",
                    ax=ax_rmse,
                    order=["BL_SOTA_tf", "BL_SOTA_tfidf", "BL_SOTA_le",
                           "Gap", "G-means",
                           "E_HAC_C", "E_OPTICS", "E_SPKMeans",
                           "E_COP_KMeans"],
                    palette=["#84d674", "#84d674", "#84d674",
                             "#fafafc", "#fafafc",
                             "#3c6cf0", "#3c6cf0", "#3c6cf0", "#3c6cf0"],
                    edgecolor=".2")

        # Rotate the x axes
        for ax in fig_rmse.axes:
            plt.sca(ax)
            plt.xticks(rotation=90)
        self.show_values_on_bars(ax_rmse, size=21)
        plt.tight_layout()
        plt.show()
        plt.close()

        self.figs.update({f"k_deviation{key_suff}": fig,
                          f"k_rmse{key_suff}": fig_rmse})
        return k_vals, df_rmse

    def visualise_cluster_sizes_hist(self,
                                     train_path: str,
                                     test_path: str,
                                     train_only: bool = False):
        freq = []
        with Tools.scan_directory(f"{train_path}\\truth") as pss:
            for ps in pss:
                filepath = f"{ps.path}\\clustering.json"
                labels = Tools.load_true_clusters_into_vector(filepath)
                freq.extend(labels.value_counts())
        if not train_only:
            with Tools.scan_directory(f"{test_path}\\truth") as pss:
                for ps in pss:
                    filepath = f"{ps.path}\\clustering.json"
                    labels = Tools.load_true_clusters_into_vector(filepath)
                    freq.extend(labels.value_counts())

        hist_data = pd.Series(freq, name="Clusters Sizes").value_counts()

        fig, ax = plt.subplots(clear=True, figsize=self.square)

        sns.barplot(x=hist_data.index, y=hist_data.values,
                    ci="sd", color="black", ax=ax)
        ax.set(xlabel="Cluster Size", ylabel="Frequency")
        ax.set_title(f"# Clusters = {hist_data.sum()}")
        plt.tight_layout()
        plt.show()
        plt.close()

        self.figs.update({"Cluster_size_histogram": fig})
        return fig, hist_data

    def visualise_nemenyi_post_hoc(self,
                                   b3f_path: str,
                                   a: float,
                                   ari_path: str = None):
        if ari_path is None:
            df_b3f = pd.read_csv(b3f_path, low_memory=True, index_col=0)
            df_b3f = df_b3f.sort_index().sort_index(axis=1)
            fig_nemenyi, ax_nemenyi = plt.subplots(nrows=1,
                                                   ncols=1,
                                                   clear=True,
                                                   sharey="row",
                                                   figsize=self.large_square)
            sns.heatmap((abs(df_b3f)), annot=True, center=a,
                        cbar=False, linewidths=0.5, fmt="0.3f", square=True,
                        cmap=ListedColormap(["#68b025", "#dadce0"]),
                        ax=ax_nemenyi)
            ax_nemenyi.set_title("bcubed_fscore")
        else:
            df_b3f = pd.read_csv(b3f_path, low_memory=True, index_col=0)
            df_b3f = df_b3f.sort_index().sort_index(axis=1)
            df_ari = pd.read_csv(ari_path, low_memory=True, index_col=0)
            df_ari = df_ari.sort_index().sort_index(axis=1)
            fig_nemenyi, ax_nemenyi = plt.subplots(nrows=1,
                                                   ncols=2,
                                                   clear=True,
                                                   sharey="row",
                                                   figsize=self.large_square)

            sns.heatmap((abs(df_b3f)), annot=True, center=a,
                        cbar=False, linewidths=0.5, fmt="0.3f", square=True,
                        cmap=ListedColormap(["#68b025", "#dadce0"]),
                        ax=ax_nemenyi[0])
            ax_nemenyi[0].set_title("bcubed_fscore")
            sns.heatmap((abs(df_ari)), annot=True, center=a,
                        cbar=False, linewidths=0.5, fmt="0.3f", square=True,
                        cmap=ListedColormap(["#68b025", "#dadce0"]),
                        ax=ax_nemenyi[1])
            ax_nemenyi[1].set_title("ari")

        plt.tight_layout()
        plt.show()
        plt.close()

        self.figs.update({"Nemenyi_post_hoc": fig_nemenyi})

        if ari_path:
            return fig_nemenyi, df_b3f, df_ari
        else:
            return fig_nemenyi, df_b3f

    def serialise_figs(self,
                       out_dir: str = r".\__outputs__\charts",
                       name: str = "Charts",
                       dpi: int = 600,
                       charts_format: str = "pdf",
                       flush: bool = False):
        if self.figs is None:
            print("No cached figures were found.")
            return

        # Append date:
        ts = pd.to_datetime("now").strftime("%Y%m%d%H%S")
        out_dir = f"{out_dir}\\{name}_{ts}"
        Tools.initialise_directories(out_dir)
        for fk in self.figs.keys():
            figure = self.figs[fk]
            figure.savefig(fname=f"{out_dir}\\{fk}.{charts_format}",
                           dpi=dpi,
                           format=charts_format,
                           transparent=False,
                           bbox_inches="tight")
        # Clear the cache
        if flush:
            self.figs = None


if __name__ == "__main__":
    print("Starting visualisations..")

# =============================================================================
#     # Locate the files to analyse:
#     # First the training data
#     sparse = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
#               r"\authorship_clustering_code_repo\__outputs__"
#               r"\results_20190924_213257_training_sparse_common.csv")
#     dense = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
#              r"\authorship_clustering_code_repo\__outputs__"
#              r"\results_20190924_213715_training_dense_common.csv")
#     neutral = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
#                r"\authorship_clustering_code_repo\__outputs__"
#                r"\results_20190924_211938_training_neutral_common.csv")
#
#     k_sparse = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
#                 r"\authorship_clustering_code_repo\__outputs__"
#                 r"\k_trend_20190924_213257_training_sparse_common.csv")
#     k_dense = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
#                r"\authorship_clustering_code_repo\__outputs__"
#                r"\k_trend_20190924_213715_training_dense_common.csv")
#     k_neutral = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
#                  r"\authorship_clustering_code_repo\__outputs__"
#                  r"\k_trend_20190924_211938_training_neutral_common.csv")
#
#     trace_sparse = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Datasets"
#                     r"\pan17_train\problem015"
#                     r"\hdp_lss_0.30_0.10_0.10_common_True"
#                     r"\state.log")
#     trace_dense = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Datasets"
#                    r"\pan17_train\problem015"
#                    r"\hdp_lss_0.80_1.50_1.50_common_True"
#                    r"\state.log")
#     trace_neutral = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Datasets"
#                      r"\pan17_train\problem015"
#                      r"\hdp_lss_0.50_1.00_1.00_common_True"
#                      r"\state.log")
# =============================================================================

    # Analyse charts and construct the cached pool:
    params = {"font.size": 21, "axes.labelsize": 21, "legend.fontsize": 16.0,
              "axes.titlesize": 21,
              "xtick.labelsize": 24, "ytick.labelsize": 21}
    vis = Visualiser(rc=params)

# =============================================================================
#     vis.analyse_results(concise=True,
#                         test_style=False,
#                         sparse_path=sparse,
#                         dense_path=dense,
#                         neutral_path=neutral,
#                         key_suff="_training_est_k")
#
#     vis.analyse_k_trends(concise=True,
#                          k_vals_path=k_sparse,
#                          key_suff="_training_sparse")
#     vis.plot_gibbs_trace(state_path=trace_sparse,
#                          key_suff="_training_sparse")
#
#     vis.analyse_k_trends(concise=True,
#                          k_vals_path=k_dense,
#                          key_suff="_training_dense")
#     vis.plot_gibbs_trace(state_path=trace_dense,
#                          key_suff="_training_dense")
#
#     vis.analyse_k_trends(concise=True,
#                          k_vals_path=k_neutral,
#                          key_suff="_training_neutral")
#     vis.plot_gibbs_trace(state_path=trace_neutral,
#                          key_suff="_training_neutral")
# =============================================================================

    # Now the test data
    sparse = (r".\__outputs__\TESTS"
              r"\results_20200331_152943_final_sparse.csv")
    # dense = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
    #          r"\authorship_clustering_code_repo\__outputs__\TESTS"
    #          r"\results_20191028_204221_final_dense.csv")
    # neutral = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
    #            r"\authorship_clustering_code_repo\__outputs__\TESTS"
    #            r"\results_20191028_201333_final_neutral.csv")

    k_sparse = (r".\__outputs__\TESTS"
                r"\k_trend_20200331_152943_final_sparse.csv")
    # k_dense = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
    #            r"\authorship_clustering_code_repo\__outputs__\TESTS"
    #            r"\k_trend_20191028_204221_final_dense.csv")
    # k_neutral = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
    #              r"\authorship_clustering_code_repo\__outputs__\TESTS"
    #              r"\k_trend_20191028_201334_final_neutral.csv")

    trace_sparse = (r"..\..\Datasets"
                    r"\pan17_test\problem015\lss_0.30_0.10_0.10_common_True"
                    r"\state.log")
    # trace_dense = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Datasets"
    #                r"\pan17_test\problem015\lss_0.80_1.50_1.50_common_True"
    #                r"\state.log")
    # trace_neutral = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Datasets"
    #                  r"\pan17_test\problem015\lss_0.50_1.00_1.00_common_True"
    #                  r"\state.log")

    sparse_true_k = (r".\__outputs__\TESTS"
                     r"\results_20200331_153147_final_trueK_sparse.csv")
    # dense_true_k = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
    #                 r"\authorship_clustering_code_repo\__outputs__\TESTS"
    #                 r"\results_20191029_142920_final_trueK_dense.csv")
    # neutral_true_k = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
    #                   r"\authorship_clustering_code_repo\__outputs__\TESTS"
    #                   r"\results_20191029_142810_final_trueK_neutral.csv")

    vis.visualise_cluster_sizes_hist(
            train_only=False,
            train_path=(r"..\..\Datasets\pan17_train"),
            test_path=(r"..\..\Datasets\pan17_test")
            )

    vis.analyse_results(concise=True,
                        test_style=True,
                        sparse_path=sparse,
                        dense_path=None,
                        neutral_path=None,
                        key_suff="_est_k")

    vis.analyse_k_trends(concise=True,
                         k_vals_path=k_sparse,
                         key_suff="_sparse")
    vis.plot_gibbs_trace(state_path=trace_sparse,
                         key_suff="_sparse")

#    vis.analyse_k_trends(concise=True,
#                         k_vals_path=k_dense,
#                         key_suff="_dense")
#    vis.plot_gibbs_trace(state_path=trace_dense,
#                         key_suff="_dense")
#
#    vis.analyse_k_trends(concise=True,
#                         k_vals_path=k_neutral,
#                         key_suff="_neutral")
#    vis.plot_gibbs_trace(state_path=trace_neutral,
#                         key_suff="_neutral")

    vis.analyse_true_k_results(true_path=sparse_true_k,
                               est_path=sparse,
                               key_suff="_true_k")

#    vis.visualise_nemenyi_post_hoc(
#            b3f_path=(r"D:\College\DKEM\Thesis\AuthorshipClustering"
#                      r"\Code\authorship_clustering_code_repo\__outputs__"
#                      r"\TESTS\Friedman_Nemenyi_B3F_a_0.0500.csv"),
#            a=0.05)
    # Serialise the cached pool to disk
    # vis.serialise_figs(charts_format="eps")
