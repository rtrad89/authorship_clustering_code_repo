# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 02:36:07 2019

@author: RTRAD
"""
from lss_modeller import LssHdpModeller
from clustering import Clusterer
from aiders import Tools
from sklearn.preprocessing import normalize
import warnings

warnings.filterwarnings(action="ignore")  # Supress warning for this code file


def problem_set_run(problem_set_id: int,
                    n_clusters: int,
                    seed: int,
                    infer_lss: bool = False,
                    verbose: bool = False):
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
            hdp_seed=seed,
            hdp_sample_hyper="no",
            word_grams=1,
            verbose=verbose)

    # Infer the BoW and LSS representations of the documents
    try:
        # Load, project and visualise the data
        plain_docs, bow_rep_docs, lss_rep_docs = Modeller.get_corpus_lss(
                infer_lss)

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
                            desired_n_clusters=n_clusters)

        # spkmeans normalises by default using the l2 norm
        norm_spk_pred, norm_spk_evals = clu_lss.eval_cluster_spherical_kmeans()

        # Normalise the data for other algorithms
        clu_lss.set_data(normalize(lss_rep_docs, norm="l2"))

        norm_dbscan_pred, norm_dbscan_evals = clu_lss.eval_cluster_dbscan(
                epsilon=0.05, min_pts=2)
        norm_hdbscan_pred, norm_hdbscan_evals = clu_lss.eval_cluster_hdbscan()
        norm_ms_pred, norm_ms_evals = clu_lss.eval_cluster_mean_shift()
        norm_xm_pred, norm_xm_evals = clu_lss.eval_cluster_xmeans()
        nhac_complete_pred, nhac_complete_evals = clu_lss.eval_cluster_hac()
        nhac_s_pred, nhac_s_evals = clu_lss.eval_cluster_hac(linkage="single")
        nhac_a_pred, nhac_a_evals = clu_lss.eval_cluster_hac(linkage="average")
        nhdp_pred, nhdp_evals = clu_lss.eval_cluster_hdp()
        ntrue_pred, ntrue_evals = clu_lss.eval_true_clustering()

        # Return the results:
        return (Tools.form_problemset_result_dictionary(
                dictionaries=[
                        norm_spk_evals, norm_dbscan_evals,
                        norm_hdbscan_evals, norm_ms_evals, norm_xm_evals,
                        nhac_complete_evals, nhac_s_evals, nhac_a_evals,
                        nhdp_evals, ntrue_evals
                        ],
                identifiers=["SPKMEANS", "DBSCAN", "HDBSCAN",
                             "MeanShift", "XMEANS", "HAC_COMPLETE",
                             "HAC_SINGLE", "HAC_AVERAGE", "HDP", "TRUE"],
                problem_set=problem_set_id),
                ground_truth,
                lss_rep_docs,
                plain_docs,
                clu_lss)

    except FileNotFoundError:
        print("Please run HDP on these data first.")


if __name__ == "__main__":
    # A list of problem sets where the random seed 33 wasn't compatible with
    # hyper sampling being on, so results were produced but not saved
    problematics = [15, 30, 45, 51, 56, 59, 60]
    problemsets_results = []
    print("▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\n")
    for ps in range(1, 61):
        print(f"Executing on problem set ► {ps:03d} ◄ ..")
        ps_result, l, lss, plain, clu = problem_set_run(
                problem_set_id=ps,
                infer_lss=True,
                hyper_sampling=False,
                # Emperically specify a random seed that's compatible with
                # hyper sampling and certain problem sets due to a bug in HDP
                # as it seems. However, the seeds would be consistant across
                # runs and yield comparable results for our experiments
                # (comparing different runs of HDP on a problem set)
                seed=max(33, 70*(ps == 41)) + (3 * (ps in problematics)),
                verbose=False,
                n_clusters=0)
        problemsets_results.append(ps_result)
        print("\n▬▬▬▬▬▬▬▬▬▬▬▬▬(Done)▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\n")
    Tools.splice_save_problemsets_dictionaries(problemsets_results)
    print("Execution finished.")
