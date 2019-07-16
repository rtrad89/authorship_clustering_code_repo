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


# Define an LSS modeller to represent documents in LSS non-sparse space
# HDP with Gibbs sampler is being used as is from:
#   https://github.com/blei-lab/hdp
problem_nbr = "01"
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
plain_docs, bow_rep_docs, lss_rep_docs = Modeller.get_corpus_lss(False)

# Try an HDBSCAN clustering
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
print("\n> DBSCAN Results:")
pprint(evals)

print("\n**********************************")
print("▬▬▬▬▬▬▬▬NORMALISED RESULTS▬▬▬▬▬▬▬▬\n")
print("**********************************")

# Experiment with normalised data
# spkmeans normalises by default using the l2 norm
npred, nevals = clu_lss.eval_cluster_sphirical_kmeans(k=None)
print("\n> Sphirical K-Means Results:")
pprint(nevals)
# Normalise the data for other algorithms
clu_lss.set_data(DataFrame(normalize(lss_rep_docs, norm="l2")))
norm_pred, norm_evals = clu_lss.eval_cluster_dbscan(epsilon=0.05, min_pts=2)
print("\n> DBSCAN Results")
pprint(norm_evals)

