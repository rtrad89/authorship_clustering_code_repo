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
from matplotlib.colors import ListedColormap


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
                 landscape_size: tuple = (15, 9),
                 rectangle_size: tuple = (15, 5),
                 double_square_size: tuple = (15, 7.5)):
        # Set seaborn defaults
        sns.set(font_scale=scale)
        sns.set_style(style)
        self.error_width = error_width
        self.error_cap_size = errorcap_size
        self.single = single_size
        self.square = square_size
        self.portrait = portrait_size
        self.landscape = landscape_size
        self.rectangle = rectangle_size
        self.double_square = double_square_size
        self.figs = {}

    def show_values_on_bars(self, axs):
        def _show_on_single_plot(ax):
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = .5 * (p.get_y() + p.get_height())
                value = f"{p.get_height():0.3f}"
                ax.text(_x, _y, value, ha="center",
                        fontdict={"color": "black",
                                  "weight": "bold",
                                  "size": 10})

        if isinstance(axs, pd.np.ndarray):
            for idx, ax in pd.np.ndenumerate(axs):
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

    def analyse_results(self,
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
                            ["HDBSCAN", "HAC_Single", "HAC_Average"])]
            df_sparse_res = df_sparse_res[
                    ~df_sparse_res.algorithm.isin(
                            ["HDBSCAN", "HAC_Single", "HAC_Average"])]
            df_dense_res = df_dense_res[
                    ~df_dense_res.algorithm.isin(
                            ["HDBSCAN", "HAC_Single", "HAC_Average"])]

        if test_style:
            df_all = df_sparse_res
        else:
            df_all = pd.concat(
                    [df_neutral_res, df_sparse_res, df_dense_res],
                    axis=0,
                    keys=["Neutral", "Sparse", "Dense"]
                    ).reset_index(level=0).rename(
                            columns={"level_0": "topics_prior"})

        algo_order = ["BL_r", "BL_s", "BL_SOTA",
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

    def analyse_k_trends(self,
                         concise: bool,
                         k_vals_path: str,
                         key_suff: str):
        if concise:
            k_vals = pd.read_csv(k_vals_path, low_memory=False,
                                 usecols=["Est_k", "Gap",
                                          "G-means", "Hac_c", "OPTICS",
                                          "TRUE"])
        else:
            k_vals = pd.read_csv(k_vals_path, low_memory=False, index_col=0)

        # FOR TRIAL
#        k_vals["ex_bic_avg"] = .5 * (k_vals["gap"] + k_vals["gmeans"])
#        k_vals["ex_bic_hac_avg"] = (k_vals["gap"] + k_vals["gmeans"]
#                                    + k_vals["hac_c"]) / 3

        fig, ax = plt.subplots(nrows=1,
                               ncols=3,
                               clear=True,
                               figsize=self.rectangle,
                               sharex="col")

        sns.scatterplot(x="TRUE", y="Est_k", color=".0",
                        marker="X",
                        ax=ax[0],
                        data=k_vals)
        sns.scatterplot(x="TRUE", y="Hac_c", color=".0",
                        marker="P",
                        ax=ax[1],
                        data=k_vals)
        sns.scatterplot(x="TRUE", y="OPTICS", color=".0",
                        marker="^",
                        ax=ax[2],
                        data=k_vals)

        if not concise:
            sns.scatterplot(x="TRUE", y="Hac_s", color=".0",
                            marker="s",
                            ax=ax[1, 1],
                            data=k_vals)
            sns.scatterplot(x="TRUE", y="Hac_a", color=".0",
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
        sota_pred = Tools.get_sota_est_k(
                output_path=(r"D:\College\DKEM\Thesis\AuthorshipClustering"
                             r"\Code\clusterPAN2017-master\output_LogEnt"))
        rmse.update({"BL_SOTA": Tools.calc_rmse(k_vals["TRUE"], sota_pred)})

        df_rmse = pd.DataFrame(data=rmse, index=[0]).T
        df_rmse.columns = ["RMSE"]

        fig_rmse, ax_rmse = plt.subplots(clear=True, figsize=self.single)
        sns.barplot(x=df_rmse.index, y="RMSE", data=df_rmse, ci="sd",
                    ax=ax_rmse)

        # Rotate the x axes
        for ax in fig_rmse.axes:
            plt.sca(ax)
            plt.xticks(rotation=90)
        self.show_values_on_bars(ax_rmse)
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

        fig, ax = plt.subplots(clear=True, figsize=self.single)

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
                                   ari_path: str):
        df_b3f = pd.read_csv(b3f_path, low_memory=True, index_col=0)
        df_b3f = df_b3f.sort_index().sort_index(axis=1)
        df_ari = pd.read_csv(ari_path, low_memory=True, index_col=0)
        df_ari = df_ari.sort_index().sort_index(axis=1)

        fig_nemenyi, ax_nemenyi = plt.subplots(nrows=1,
                                               ncols=2,
                                               clear=True,
                                               sharey="row",
                                               figsize=self.double_square)

        sns.heatmap((abs(df_b3f)), annot=True, center=0.025,
                    cbar=False, linewidths=0.5, fmt="0.3f", square=True,
                    cmap=ListedColormap(["#68b025", "#dadce0"]),
                    ax=ax_nemenyi[0])
        ax_nemenyi[0].set_title("bcubed_fscore")
        sns.heatmap((abs(df_ari)), annot=True, center=0.025,
                    cbar=False, linewidths=0.5, fmt="0.3f", square=True,
                    cmap=ListedColormap(["#68b025", "#dadce0"]),
                    ax=ax_nemenyi[1])
        ax_nemenyi[1].set_title("ari")

        plt.tight_layout()
        plt.show()
        plt.close()

        self.figs.update({"Nemenyi_post_hoc": fig_nemenyi})
        return fig_nemenyi, df_b3f, df_ari

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
                           transparent=False,  # EPS doesn't support it already
                           metadata={"Author": "Rafi Trad"})
        # Clear the cache
        if flush:
            self.figs = None


if __name__ == "__main__":
    print("Starting visualisations..")

    # Locate the files to analyse:
    # First the training data
    sparse = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
              r"\authorship_clustering_code_repo\__outputs__"
              r"\results_20190924_213257_training_sparse_common.csv")
    dense = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
             r"\authorship_clustering_code_repo\__outputs__"
             r"\results_20190924_213715_training_dense_common.csv")
    neutral = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
               r"\authorship_clustering_code_repo\__outputs__"
               r"\results_20190924_211938_training_neutral_common.csv")

    k_sparse = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
                r"\authorship_clustering_code_repo\__outputs__"
                r"\k_trend_20190924_213257_training_sparse_common.csv")
    k_dense = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
               r"\authorship_clustering_code_repo\__outputs__"
               r"\k_trend_20190924_213715_training_dense_common.csv")
    k_neutral = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
                 r"\authorship_clustering_code_repo\__outputs__"
                 r"\k_trend_20190924_211938_training_neutral_common.csv")

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
    vis = Visualiser(scale=1.2)

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
# =============================================================================

    vis.analyse_k_trends(concise=True,
                         k_vals_path=k_neutral,
                         key_suff="_training_neutral")
# =============================================================================
#     vis.plot_gibbs_trace(state_path=trace_neutral,
#                          key_suff="_training_neutral")
# =============================================================================

    # Now the test data
    sparse = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
              r"\authorship_clustering_code_repo\__outputs__\TESTS"
              r"\results_20190923_234123_final_sparse.csv")
    dense = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
             r"\authorship_clustering_code_repo\__outputs__\TESTS"
             r"\results_20190923_231000_final_dense.csv")
    neutral = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
               r"\authorship_clustering_code_repo\__outputs__\TESTS"
               r"\results_20190923_225951_final_neutral.csv")

    k_sparse = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
                r"\authorship_clustering_code_repo\__outputs__\TESTS"
                r"\k_trend_20190923_234123_final_sparse.csv")
    k_dense = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
               r"\authorship_clustering_code_repo\__outputs__\TESTS"
               r"\k_trend_20190923_231000_final_dense.csv")
    k_neutral = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
                 r"\authorship_clustering_code_repo\__outputs__\TESTS"
                 r"\k_trend_20190923_225952_final_neutral.csv")

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
                     r"\results_20190923_235606_final_trueK_sparse.csv")
    dense_true_k = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
                    r"\authorship_clustering_code_repo\__outputs__\TESTS"
                    r"\results_20190923_235117_final_trueK_dense.csv")
    neutral_true_k = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
                      r"\authorship_clustering_code_repo\__outputs__\TESTS"
                      r"\results_20190923_234615_final_trueK_neutral.csv")

# =============================================================================
#     vis.visualise_cluster_sizes_hist(
#             train_only=False,
#             train_path=(r"D:\College\DKEM\Thesis\AuthorshipClustering"
#                         r"\Datasets\pan17_train"),
#             test_path=(r"D:\College\DKEM\Thesis\AuthorshipClustering"
#                        r"\Datasets\pan17_test")
#             )
# 
#     vis.analyse_results(concise=True,
#                         test_style=True,
#                         sparse_path=sparse,
#                         dense_path=dense,
#                         neutral_path=neutral,
#                         key_suff="_est_k")
# =============================================================================

    vis.analyse_k_trends(concise=True,
                         k_vals_path=k_sparse,
                         key_suff="_sparse")
# =============================================================================
#     vis.plot_gibbs_trace(state_path=trace_sparse,
#                          key_suff="_sparse")
# 
#     vis.analyse_k_trends(concise=True,
#                          k_vals_path=k_dense,
#                          key_suff="_dense")
#     vis.plot_gibbs_trace(state_path=trace_dense,
#                          key_suff="_dense")
# 
#     vis.analyse_k_trends(concise=True,
#                          k_vals_path=k_neutral,
#                          key_suff="_neutral")
#     vis.plot_gibbs_trace(state_path=trace_neutral,
#                          key_suff="_neutral")
# 
#     vis.analyse_results(concise=True,
#                         test_style=True,
#                         sparse_path=sparse_true_k,
#                         dense_path=dense_true_k,
#                         neutral_path=neutral_true_k,
#                         key_suff="_true_k")
#     vis.visualise_nemenyi_post_hoc(
#             b3f_path=(r"D:\College\DKEM\Thesis\AuthorshipClustering"
#                       r"\Code\authorship_clustering_code_repo\__outputs__"
#                       r"\TESTS\Friedman_Nemenyi_B3F_a_0.0250.csv"),
#             ari_path=(r"D:\College\DKEM\Thesis\AuthorshipClustering"
#                       r"\Code\authorship_clustering_code_repo\__outputs__"
#                       r"\TESTS\Friedman_Nemenyi_ARI_a_0.0250.csv")
#             )
#     # Serialise the cached pool to disk
#     vis.serialise_figs(charts_format="eps")
# =============================================================================
    # Run Friedman-Nemenyi test with Bonferroni correction for multiple tests
    # since the dataset is the same
    print(Tools.friedman_nemenyi_bonferroni_tests(
            data_path=sparse, save_outputs=False))
