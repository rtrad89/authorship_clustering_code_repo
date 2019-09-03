# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 11:59:09 2019
A class which caters to visualisation

@author: RTRAD
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from aiders import Tools


class Visualiser():
    # Predefined styles values
    style_whitegrid = "whitegrid"
    style_darkgrid = "darkgrid"
    style_dark = "dark"
    style_white = "white"
    style_ticks = "ticks"

    def __init__(self,
                 scale: float,
                 style: str = "darkgrid",
                 error_width: float = 1.0,
                 errorcap_size: float = 0.05,
                 single_size: tuple = (5, 5),
                 square_size: tuple = (9, 9),
                 portrait_size: tuple = (9, 15),
                 landscape_size: tuple = (15, 9)):
        # Set seaborn defaults
        sns.set(font_scale=scale)
        sns.set_style(style)
        self.error_width = error_width
        self.error_cap_size = errorcap_size
        self.single = single_size
        self.square = square_size
        self.portrait = portrait_size
        self.landscape = landscape_size
        self.figs = {}

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

    def analyse_results(self,
                        concise: bool,
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
                            ["HDBSCAN", "HAC_Single", "HAC_Average"])]
            df_sparse_res = df_sparse_res[
                    ~df_sparse_res.algorithm.isin(
                            ["HDBSCAN", "HAC_Single", "HAC_Average"])]
            df_dense_res = df_dense_res[
                    ~df_dense_res.algorithm.isin(
                            ["HDBSCAN", "HAC_Single", "HAC_Average"])]

        df_all = pd.concat(
                [df_neutral_res, df_sparse_res, df_dense_res],
                axis=0,
                keys=["Neutral", "Sparse", "Dense"]
                ).reset_index(level=0).rename(
                        columns={"level_0": "topics_prior"})

        fig_overall, ax_overall = plt.subplots(nrows=2,
                                               ncols=1,
                                               clear=True,
                                               figsize=self.portrait)
        sns.barplot(x="algorithm", y="bcubed_fscore", hue="topics_prior",
                    data=df_all,
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    ax=ax_overall[0])
        sns.barplot(x="algorithm", y="ari", hue="topics_prior",
                    data=df_all,
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    ax=ax_overall[1])

        # Rotate the x labels:
        for ax in fig_overall.axes:
            plt.sca(ax)
            ax.set_xlabel("")
            plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        plt.close()

        fig_genre_lang, ax_genre_lang = plt.subplots(nrows=2,
                                                     ncols=2,
                                                     clear=True,
                                                     figsize=self.landscape)
        sns.barplot(x="language", y="bcubed_fscore", hue="algorithm",
                    data=df_sparse_res,
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    ax=ax_genre_lang[0, 0])
        sns.barplot(x="genre", y="bcubed_fscore", hue="algorithm",
                    data=df_sparse_res,
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    ax=ax_genre_lang[0, 1])
        sns.barplot(x="language", y="ari", hue="algorithm",
                    data=df_sparse_res,
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    ax=ax_genre_lang[1, 0])
        sns.barplot(x="genre", y="ari", hue="algorithm",
                    data=df_sparse_res,
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    ax=ax_genre_lang[1, 1])

        plt.legend(loc="lower left")
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
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    ax=ax_comb_genre_lang[0, 0])
        ax_comb_genre_lang[0, 0].set_title("English")
        sns.barplot(x="algorithm", y="bcubed_fscore", hue="genre",
                    data=df_sparse_res[df_sparse_res.language == "nl"],
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    ax=ax_comb_genre_lang[0, 1])
        ax_comb_genre_lang[0, 1].set_title("Dutch")
        sns.barplot(x="algorithm", y="bcubed_fscore", hue="genre",
                    data=df_sparse_res[df_sparse_res.language == "gr"],
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    ax=ax_comb_genre_lang[0, 2])
        ax_comb_genre_lang[0, 2].set_title("Greek")

        sns.barplot(x="algorithm", y="ari", hue="genre",
                    data=df_sparse_res[df_sparse_res.language == "en"],
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    ax=ax_comb_genre_lang[1, 0])
        sns.barplot(x="algorithm", y="ari", hue="genre",
                    data=df_sparse_res[df_sparse_res.language == "nl"],
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    ax=ax_comb_genre_lang[1, 1])
        sns.barplot(x="algorithm", y="ari", hue="genre",
                    data=df_sparse_res[df_sparse_res.language == "gr"],
                    errwidth=self.error_width,
                    capsize=self.error_cap_size,
                    ax=ax_comb_genre_lang[1, 2])

        for ax in fig_comb_genre_lang.axes:
            plt.sca(ax)
            ax.set_xlabel("")
            plt.xticks(rotation=90)
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.show()
        plt.close()

        self.figs.update(
                {f"Results_Overall{key_suff}": fig_overall,
                 f"Results_genre_lang{key_suff}": fig_genre_lang,
                 f"Results_combined_genre_lang{key_suff}": fig_comb_genre_lang
                 })

        return df_all

    def analyse_k_trends(self,
                         concise: bool,
                         k_vals_path: str,
                         key_suff: str):
        if concise:
            k_vals = pd.read_csv(k_vals_path, low_memory=False,
                                 usecols=["est_k", "bic", "gap",
                                          "gmeans", "hac_c", "est_avg_c",
                                          "true"])
        else:
            k_vals = pd.read_csv(k_vals_path, low_memory=False, index_col=0)

        # FOR TRIAL
#        k_vals["ex_bic_avg"] = .5 * (k_vals["gap"] + k_vals["gmeans"])
#        k_vals["ex_bic_hac_avg"] = (k_vals["gap"] + k_vals["gmeans"]
#                                    + k_vals["hac_c"]) / 3

        fig, ax = plt.subplots(nrows=2,
                               ncols=3,
                               clear=True,
                               figsize=self.landscape,
                               sharex="col")

        sns.scatterplot(x="true", y="est_k", color=".0",
                        marker="X",
                        ax=ax[0, 0],
                        data=k_vals)

        sns.scatterplot(x="true", y="hac_c", color=".0",
                        marker="P",
                        ax=ax[1, 0],
                        data=k_vals)
        sns.scatterplot(x="true", y="est_avg_c", color=".0",
                        marker="^",
                        ax=ax[0, 1],
                        data=k_vals)

        if not concise:
            sns.scatterplot(x="true", y="hac_s", color=".0",
                            marker="s",
                            ax=ax[1, 1],
                            data=k_vals)
            sns.scatterplot(x="true", y="hac_a", color=".0",
                            marker="o",
                            ax=ax[1, 2],
                            data=k_vals)

        plt.tight_layout()
        plt.show()
        plt.close()

        # Plot the RMSE trends
        rmse = {}
        for col in k_vals.columns:
            if col in ["true"]:
                continue
            rmse.update({col: Tools.calc_rmse(k_vals.true, k_vals[col])})

        df_rmse = pd.DataFrame(data=rmse, index=[0]).T
        df_rmse.columns = ["RMSE"]

        fig_rmse, ax_rmse = plt.subplots(clear=True, figsize=self.single)
        sns.barplot(x=df_rmse.index, y="RMSE", data=df_rmse, ax=ax_rmse)

        # Rotate the x axes
        for ax in fig_rmse.axes:
            plt.sca(ax)
            plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
        plt.close()

        self.figs.update({f"k_deviation{key_suff}": fig,
                          f"k_rmse{key_suff}": fig_rmse})
        return k_vals, df_rmse

    def serialise_figs(self,
                       out_dir: str = r".\__outputs__\charts",
                       name: str = "Charts",
                       dpi: int = 600,
                       charts_format: str = "pdf",
                       flush: bool = True):
        if vis.figs is None:
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
                           transparent=False,  # EPS doesn't support it already
                           metadata={"Author": "Rafi Trad"})
        # Clear the cache
        if flush:
            vis.figs = None


if __name__ == "__main__":
    # Locate the files to analyse:
    # First the training data
    sparse = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
              r"\authorship_clustering_code_repo\__outputs__"
              r"\results_20190902_235056_training_sparse_common.csv")
    dense = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
             r"\authorship_clustering_code_repo\__outputs__"
             r"\results_20190902_235509_training_dense_common.csv")
    neutral = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
               r"\authorship_clustering_code_repo\__outputs__"
               r"\results_20190902_233804_training_neutral_common.csv")

    k_sparse = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
                r"\authorship_clustering_code_repo\__outputs__"
                r"\k_trend_20190902_235056_training_sparse_common.csv")
    k_dense = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
               r"\authorship_clustering_code_repo\__outputs__"
               r"\k_trend_20190902_235509_training_dense_common.csv")
    k_neutral = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
                 r"\authorship_clustering_code_repo\__outputs__"
                 r"\k_trend_20190902_233804_training_neutral_common.csv")

    trace_sparse = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Datasets"
                    r"\pan17_train\problem015"
                    r"\hdp_lss_0.30_0.10_0.10_common_True"
                    r"\state.log")
    trace_dense = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Datasets"
                   r"\pan17_train\problem015"
                   r"\hdp_lss_0.80_1.50_1.50_common_True"
                   r"\state.log")
    trace_neutral = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Datasets"
                     r"\pan17_train\problem015"
                     r"\hdp_lss_0.50_1.00_1.00_common_True"
                     r"\state.log")

    # Analyse charts and construct the cached pool:
    vis = Visualiser(scale=1.25)

    vis.analyse_results(concise=True,
                        sparse_path=sparse,
                        dense_path=dense,
                        neutral_path=neutral,
                        key_suff="_training_est_k")

    vis.analyse_k_trends(concise=True,
                         k_vals_path=k_sparse,
                         key_suff="_training_sparse")
    vis.plot_gibbs_trace(state_path=trace_sparse,
                         key_suff="_training_sparse")

    vis.analyse_k_trends(concise=True,
                         k_vals_path=k_dense,
                         key_suff="_training_dense")
    vis.plot_gibbs_trace(state_path=trace_dense,
                         key_suff="_training_dense")

    vis.analyse_k_trends(concise=True,
                         k_vals_path=k_neutral,
                         key_suff="_training_neutral")
    vis.plot_gibbs_trace(state_path=trace_neutral,
                         key_suff="_training_neutral")

    # Now the test data
    sparse = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
              r"\authorship_clustering_code_repo\__outputs__\TESTS"
              r"\results_20190903_003327_final_sparse.csv")
    dense = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
             r"\authorship_clustering_code_repo\__outputs__\TESTS"
             r"\results_20190903_001458_final_dense.csv")
    neutral = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
               r"\authorship_clustering_code_repo\__outputs__\TESTS"
               r"\results_20190903_000722_final_neutral.csv")

    k_sparse = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
                r"\authorship_clustering_code_repo\__outputs__\TESTS"
                r"\k_trend_20190903_003327_final_sparse.csv")
    k_dense = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
               r"\authorship_clustering_code_repo\__outputs__\TESTS"
               r"\k_trend_20190903_001458_final_dense.csv")
    k_neutral = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
                 r"\authorship_clustering_code_repo\__outputs__\TESTS"
                 r"\k_trend_20190903_000722_final_neutral.csv")

    trace_sparse = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Datasets"
                    r"\pan17_test\problem015\lss_0.30_0.10_0.10_common_True"
                    r"\state.log")
    trace_dense = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Datasets"
                   r"\pan17_test\problem015\lss_0.80_1.50_1.50_common_True"
                   r"\state.log")
    trace_neutral = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Datasets"
                     r"\pan17_test\problem015\lss_0.50_1.00_1.00_common_True"
                     r"\state.log")

    sparse_true_k = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
                     r"\authorship_clustering_code_repo\__outputs__\TESTS"
                     r"\results_20190903_141821_final_trueK_sparse.csv")
    dense_true_k = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
                    r"\authorship_clustering_code_repo\__outputs__\TESTS"
                    r"\results_20190903_141459_final_trueK_dense.csv")
    neutral_true_k = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
                      r"\authorship_clustering_code_repo\__outputs__\TESTS"
                      r"\results_20190903_141125_final_trueK_neutral.csv")

    vis.analyse_results(concise=True,
                        sparse_path=sparse,
                        dense_path=dense,
                        neutral_path=neutral,
                        key_suff="_est_k")

    vis.analyse_k_trends(concise=True,
                         k_vals_path=k_sparse,
                         key_suff="_sparse")
    vis.plot_gibbs_trace(state_path=trace_sparse,
                         key_suff="_sparse")

    vis.analyse_k_trends(concise=True,
                         k_vals_path=k_dense,
                         key_suff="_dense")
    vis.plot_gibbs_trace(state_path=trace_dense,
                         key_suff="_dense")

    vis.analyse_k_trends(concise=True,
                         k_vals_path=k_neutral,
                         key_suff="_neutral")
    vis.plot_gibbs_trace(state_path=trace_neutral,
                         key_suff="_neutral")

    vis.analyse_results(concise=True,
                        sparse_path=sparse_true_k,
                        dense_path=dense_true_k,
                        neutral_path=neutral_true_k,
                        key_suff="_true_k")
    # Serialise the cached pool to disk
#    vis.serialise_figs(charts_format="eps")
