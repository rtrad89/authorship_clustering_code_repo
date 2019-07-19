# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 02:36:07 2019

@author: RTRAD
"""
from pandas import DataFrame
from lss_modeller import LssHdpModeller
from clustering import Clusterer
from aiders import DiskTools
from sklearn.preprocessing import normalize
from pprint import pprint
#from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Define an LSS modeller to represent documents in LSS non-sparse space
# HDP with Gibbs sampler is being used as is from:
#   https://github.com/blei-lab/hdp
problem_nbr = "03"
Modeller = LssHdpModeller(
        hdp_path=r"..\..\hdps\hdp",
        input_docs_path=r"..\..\..\Datasets\pan17_train\problem0{}".format(
                problem_nbr),
        ldac_filename=r"dummy_ldac_corpus",
        hdp_output_dir=r"hdp_lss",
        hdp_iters=10000,
        hdp_seed=13712,
        word_grams=1)

# Infer the BoW and LSS representations of the documents
try:
    # Load, project and visualise the data
    plain_docs, bow_rep_docs, lss_rep_docs = Modeller.get_corpus_lss(False)
    embedded_docs = TSNE(perplexity=5, n_iter=5000,
                         random_state=13712, metric="cosine"
                         ).fit_transform(lss_rep_docs)
    plt.scatter(embedded_docs[:, 0], embedded_docs[:, 1])
    plt.show()
    plt.close()

    # Begin Clustering Attempts
    true_labels_path = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Datasets"
                        r"\pan17_train\truth\problem0{}\clustering.json"
                        ).format(problem_nbr)

    ground_truth = DiskTools.load_true_clusters_into_vector(true_labels_path)

    clu_lss = Clusterer(dtm=lss_rep_docs,
                        true_labels=ground_truth,
                        max_nbr_clusters=len(lss_rep_docs)//2,
                        min_nbr_clusters=1,
                        min_cluster_size=2,
                        metric="cosine")

    pred, evals = clu_lss.eval_cluster_dbscan(epsilon=0.05, min_pts=2)
    h_pred, h_evals = clu_lss.eval_cluster_hdbscan()
    ms_pred, ms_evals = clu_lss.eval_cluster_mean_shift()
    xm_pred, xm_evals = clu_lss.eval_cluster_xmeans()

    print("\n> DBSCAN Results:")
    pprint(evals)

    print("\n> HDBSCAN Results:")
    pprint(h_evals)

    print("\n> Mean Shift Results:")
    pprint(ms_evals)

    print("\n> X-Means Results:")
    pprint(xm_evals)

    print("\n**********************************")
    print("▬▬▬▬▬▬▬▬NORMALISED RESULTS▬▬▬▬▬▬▬▬\n")
    print("************************************")

    # Experiment with normalised data
    # spkmeans normalises by default using the l2 norm
    norm_spk_pred, norm_spk_evals = clu_lss.eval_cluster_spherical_kmeans(
            k=None)
    print("\n> Sphirical K-Means Results:")
    pprint(norm_spk_evals)

    # Normalise the data for other algorithms
    clu_lss.set_data(DataFrame(normalize(lss_rep_docs, norm="l2")))

    norm_pred, norm_evals = clu_lss.eval_cluster_dbscan(
            epsilon=0.05, min_pts=2)
    norm_h_pred, norm_h_evals = clu_lss.eval_cluster_hdbscan()
    norm_ms_pred, norm_ms_evals = clu_lss.eval_cluster_mean_shift()
    norm_xm_pred, norm_xm_evals = clu_lss.eval_cluster_xmeans()

    print("\n> DBSCAN Results")
    pprint(norm_evals)

    print("\n> HDBSCAN Results:")
    pprint(norm_h_evals)

    print("\n> Mean Shift Results:")
    pprint(norm_ms_evals)

    print("\n> X-Means Results:")
    pprint(norm_xm_evals)

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
