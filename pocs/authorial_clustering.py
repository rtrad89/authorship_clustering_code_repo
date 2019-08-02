# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 02:36:07 2019

@author: RTRAD
"""
from lss_modeller import LssHdpModeller
from clustering import Clusterer
from aiders import Tools
from sklearn.preprocessing import normalize
# from pprint import pprint
# from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import (minmax_scale,  # 1
                                   maxabs_scale)  # 2
from pandas import DataFrame

warnings.filterwarnings(action="ignore")  # Supress warning for this code file


def problem_set_run(problem_set_id: int,
                    infer_lss: bool = False,
                    scale_features: bool = False,
                    scaling_option: int = 2,
                    hyper_sampling: bool = True,
                    verbose: bool = False,
                    visualise: bool = False):
    problem_nbr = f"{problem_set_id:03d}"
    # Define an LSS modeller to represent documents in LSS non-sparse space
    # HDP with Gibbs sampler is being used as is from:
    #   https://github.com/blei-lab/hdp

    Modeller = LssHdpModeller(
            hdp_path=r"..\..\hdps\hdp",
            input_docs_path=r"..\..\..\Datasets\pan17_train\problem{}".format(
                    problem_nbr),
            ldac_filename=r"dummy_ldac_corpus",
            hdp_output_dir=r"hdp_lss",
            hdp_iters=10000,
            hdp_seed=137,
            hdp_sample_hyper=hyper_sampling,
            word_grams=1,
            verbose=verbose)

    # Infer the BoW and LSS representations of the documents
    try:
        # Load, project and visualise the data
        plain_docs, bow_rep_docs, lss_rep_docs = Modeller.get_corpus_lss(
                infer_lss)
        if scale_features:
            # Scale the features to treat them equally in importance
            if scaling_option == 1:
                lss_rep_docs = minmax_scale(X=lss_rep_docs)
            else:
                lss_rep_docs = maxabs_scale(X=lss_rep_docs)
            # Convert the data to indexed dataframe format
            lss_rep_docs = DataFrame(data=lss_rep_docs,
                                     index=plain_docs.index)
        if visualise:
            # Reduce the dimensionality for visualisation purposes
            embedded_docs = TSNE(perplexity=5, n_iter=5000,
                                 random_state=13712, metric="cosine"
                                 ).fit_transform(lss_rep_docs)
            plt.scatter(embedded_docs[:, 0], embedded_docs[:, 1])
            plt.title(f"2-component projection of problem #{problem_nbr:}")
            plt.gcf().savefig(r"./__outputs__/projection_training_{}".format(
                    problem_nbr))
            plt.show()
        # Begin Clustering Attempts
        true_labels_path = (r"D:\College\DKEM\Thesis\AuthorshipClustering"
                            r"\Datasets\pan17_train\truth"
                            r"\problem{}\clustering.json"
                            ).format(problem_nbr)

        ground_truth = Tools.load_true_clusters_into_vector(true_labels_path)

        clu_lss = Clusterer(dtm=lss_rep_docs,
                            true_labels=ground_truth,
                            max_nbr_clusters=len(lss_rep_docs)//2,
                            min_nbr_clusters=1,
                            min_cluster_size=2,
                            metric="cosine",
                            desired_n_clusters=0)

        dbscan_pred, dbscan_evals = clu_lss.eval_cluster_dbscan(
                epsilon=0.1, min_pts=2)
        hdbscan_pred, hdbscan_evals = clu_lss.eval_cluster_hdbscan()
        ms_pred, ms_evals = clu_lss.eval_cluster_mean_shift()
        xm_pred, xm_evals = clu_lss.eval_cluster_xmeans()
        hac_complete_pred, hac_complete_evals = clu_lss.eval_cluster_HAC()
        hac_single_pred, hac_single_evals = clu_lss.eval_cluster_HAC(
                linkage="single")
        hac_average_pred, hac_average_evals = clu_lss.eval_cluster_HAC(
                linkage="average")
#    agc_pred, agc_evals = clu_lss.eval_cluster_agglomerative()

# =============================================================================
#     print("\n> DBSCAN Results:")
#     pprint(dbscan_evals)
#
#     print("\n> HDBSCAN Results:")
#     pprint(hdbscan_evals)
#
#     print("\n> Mean Shift Results:")
#     pprint(ms_evals)
#
#     print("\n> X-Means Results:")
#     pprint(xm_evals)
#
#     print("\n> HAC:")
#     pprint(hac_evals)
#
#     print("\n> Agglomerative:")
#     pprint(agc_evals)
#
#     print("\n**********************************")
#     print("▬▬▬▬▬▬▬▬NORMALISED RESULTS▬▬▬▬▬▬▬▬")
#     print("**********************************")
# =============================================================================

        # Experiment with normalised data
        # spkmeans normalises by default using the l2 norm
        norm_spk_pred, norm_spk_evals = clu_lss.eval_cluster_spherical_kmeans()

        # Normalise the data for other algorithms
        clu_lss.set_data(normalize(lss_rep_docs, norm="l2"))

        norm_dbscan_pred, norm_dbscan_evals = clu_lss.eval_cluster_dbscan(
                epsilon=0.05, min_pts=2)
        norm_hdbscan_pred, norm_hdbscan_evals = clu_lss.eval_cluster_hdbscan()
        norm_ms_pred, norm_ms_evals = clu_lss.eval_cluster_mean_shift()
        norm_xm_pred, norm_xm_evals = clu_lss.eval_cluster_xmeans()
        nhac_complete_pred, nhac_complete_evals = clu_lss.eval_cluster_HAC()
        nhac_s_pred, nhac_s_evals = clu_lss.eval_cluster_HAC(linkage="single")
        nhac_a_pred, nhac_a_evals = clu_lss.eval_cluster_HAC(linkage="average")
        # norm_agc_pred, norm_agc_evals = clu_lss.eval_cluster_agglomerative()
        plt.close()
        # Return the results:
        return (Tools.form_problemset_result_dictionary(
                dictionaries=[dbscan_evals, hdbscan_evals, ms_evals, xm_evals,
                              hac_complete_evals, hac_single_evals,
                              hac_average_evals, norm_spk_evals,
                              norm_dbscan_evals, norm_hdbscan_evals,
                              norm_ms_evals, norm_xm_evals,
                              nhac_complete_evals, nhac_s_evals, nhac_a_evals],
                l2_norms=[False, False, False, False, False, False, False,
                          True, True, True, True, True, True, True, True],
                identifiers=["DBSCAN", "HDBSCAN", "MeanShift", "XMEANS",
                             "HAC_COMPLETE", "HAC_SINGLE", "HAC_AVERAGE",
                             "SPKMEANS", "DBSCAN", "HDBSCAN",
                             "MeanShift", "XMEANS", "HAC_COMPLETE",
                             "HAC_SINGLE", "HAC_AVERAGE"],
                problem_set=problem_set_id),
                lss_rep_docs,
                plain_docs,
                clu_lss)
# =============================================================================
#     print("\n> Sphirical K-Means Results:")
#     pprint(norm_spk_evals)
#
#     print("\n> DBSCAN Results")
#     pprint(norm_dbscan_evals)
#
#     print("\n> HDBSCAN Results:")
#     pprint(norm_hdbscan_evals)
#
#     print("\n> Mean Shift Results:")
#     pprint(norm_ms_evals)
#
#     print("\n> X-Means Results:")
#     pprint(norm_xm_evals)
#
#     print("\n> HAC:")
#     pprint(norm_hac_evals)
#
#     print("\n> Agglomerative:")
#     pprint(norm_agc_evals)
# =============================================================================

# =============================================================================
#     print("\n*******************************************************")
#     print("▬▬▬▬▬▬▬▬Feature Selected Non-Normalised RESULTS▬▬▬▬▬▬▬▬\n")
#     print("*********************************************************")
#     fs_lss_df = DataFrame(VarianceThreshold(threshold=25
#                                             ).fit_transform(lss_rep_docs))
#     clu_lss.set_data(fs_lss_df)
#     # Experiment with FS data
#     # spkmeans normalises by default using the l2 norm
#     fs_npred, fs_nevals = clu_lss.eval_cluster_spherical_kmeans(k=6)
#     print("\n> Sphirical K-Means Results:")
#     pprint(fs_nevals)
#
#     fs_pred, fs_evals = clu_lss.eval_cluster_dbscan(
#             epsilon=0.05, min_pts=2)
#     print("\n> DBSCAN Results")
#     pprint(fs_evals)
# =============================================================================
    except FileNotFoundError:
        print("Please run HDP on this data first.")


if __name__ == "__main__":
    problemsets_results = []
    print("▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\n")
    for ps in range(1, 2):
        print(f"Executing on problem set ► {ps:03d} ◄ ..")
        ps_result, lss, plain, clu = problem_set_run(problem_set_id=ps,
                                                     infer_lss=False,
                                                     verbose=False)
        problemsets_results.append(ps_result)
        print("\n▬▬▬▬▬▬▬▬▬▬▬▬▬(Done)▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\n")
    Tools.splice_save_problemsets_dictionaries(problemsets_results)
    print("Execution finished.")
