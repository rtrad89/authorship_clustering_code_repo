# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 02:36:07 2019

@author: RTRAD
"""
from lss_modeller import LssHdpModeller
from aiders import Tools
from clustering import Clusterer
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
            hdp_path=r"..\hdps\hdp",
            input_docs_path=r"..\..\Datasets\pan17_train\problem{}".format(
                    problem_nbr),
            ldac_filename=r"dummy_ldac_corpus",
            hdp_output_dir=r"hdp_lss",
            hdp_iters=10000,
            hdp_seed=seed,
            hdp_sample_hyper=False,
            word_grams=1,
            verbose=verbose)

    # Infer the BoW and LSS representations of the documents
    try:
        # Load, project and visualise the data
        plain_docs, bow_rep_docs, lss_rep_docs = Modeller.get_corpus_lss(
                infer_lss)

        # Begin Clustering Attempts
        true_labels_path = (r"..\..\Datasets\pan17_train\truth"
                            r"\problem{}\clustering.json"
                            ).format(problem_nbr)

        ground_truth = Tools.load_true_clusters_into_vector(true_labels_path)

        # Normalise the data
        clu_lss = Clusterer(dtm=Tools.normalise_data(data=lss_rep_docs),
                            true_labels=ground_truth,
                            max_nbr_clusters=len(lss_rep_docs)-1,
                            min_nbr_clusters=1,
                            min_cluster_size=2,
                            metric="cosine",
                            desired_n_clusters=n_clusters)

        norm_spk_pred, norm_spk_evals = clu_lss.evaluate(
                alg_option=Clusterer.alg_spherical_k_means,
                param_init="k-means++")

        ispk_pred, ispk_evals = clu_lss.evaluate(
                alg_option=Clusterer.alg_iterative_spherical_k_means,
                param_init="k-means++")

        norm_hdbscan_pred, norm_hdbscan_evals = clu_lss.evaluate(
                alg_option=Clusterer.alg_h_dbscan)

        norm_ms_pred, norm_ms_evals = clu_lss.evaluate(
                alg_option=Clusterer.alg_mean_shift)

        norm_xm_pred, norm_xm_evals = clu_lss.evaluate(
                alg_option=Clusterer.alg_x_means)

        nhac_complete_pred, nhac_complete_evals = clu_lss.evaluate(
                alg_option=Clusterer.alg_hac,
                param_linkage="complete")

        nhac_s_pred, nhac_s_evals = clu_lss.evaluate(
                alg_option=Clusterer.alg_hac,
                param_linkage="single")

        nhac_a_pred, nhac_a_evals = clu_lss.evaluate(
                alg_option=Clusterer.alg_hac,
                param_linkage="average")

        n_optics_pred, n_optics_evals = clu_lss.evaluate(
                alg_option=Clusterer.alg_optics)

        nhdp_pred, nhdp_evals = clu_lss.eval_cluster_hdp()
        ntrue_pred, ntrue_evals = clu_lss.eval_true_clustering()

        # Return the results:
        return (Tools.form_problemset_result_dictionary(
                dictionaries=[
                        ispk_evals, norm_spk_evals, norm_hdbscan_evals,
                        norm_ms_evals, norm_xm_evals,
                        nhac_complete_evals, nhac_s_evals, nhac_a_evals,
                        n_optics_evals,
                        nhdp_evals, ntrue_evals
                        ],
                identifiers=["iSpKmeans", "SPKMEANS", "HDBSCAN",
                             "MeanShift", "XMEANS", "HAC_COMPLETE",
                             "HAC_SINGLE", "HAC_AVERAGE", "OPTICS",
                             "HDP", "TRUE"],
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
    k_vals = []
    print("▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\n")
    for ps in range(1, 61):
        print(f"Executing on problem set ► {ps:03d} ◄ ..")
        ps_result, l, lss, plain, clu = problem_set_run(
            problem_set_id=ps,
            n_clusters=None,
            # Emperically specify a random seed that's compatible with
            # hyper sampling and certain problem sets due to a bug in HDP
            # as it seems. However, the seeds would be consistant across
            # runs and yield comparable results for our experiments
            # (comparing different runs of HDP on a problem set)
            seed=max(33, 70*(ps == 41)) + (3 * (ps in problematics)),
            infer_lss=False,
            verbose=False)
        problemsets_results.append(ps_result)
        ks = clu.cand_k.copy()
        ks.append(1+max(clu.true_labels))
        k_vals.append(ks)
        print("\n▬▬▬▬▬▬▬▬▬▬▬▬▬(Done)▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\n")
    my_suffix = "_training_data"
    info_json = r"..\..\Datasets\pan17_train\info.json"
    Tools.splice_save_problemsets_dictionaries(problemsets_results,
                                               metadata_fpath=info_json,
                                               suffix=my_suffix)
    Tools.save_k_vals_as_df(k_vals=k_vals, suffix=my_suffix)
    print("Execution finished.")
