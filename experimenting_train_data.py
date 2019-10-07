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

# Sampling preference
config_sparse = "sparse"
config_neutral = "neutral"
config_dense = "dense"


def problem_set_run(problem_set_id: int,
                    n_clusters: int,
                    seed: int,
                    configuration: str,
                    drop_uncommon: bool,
                    verbose: bool,
                    infer_lss: bool = False):
    problem_nbr = f"{problem_set_id:03d}"
    # Define an LSS modeller to represent documents in LSS non-sparse space
    # HDP with Gibbs sampler is being used as is from:
    #   https://github.com/blei-lab/hdp

    # Adjust the parameters according to the preference
    if configuration == config_sparse:
        eta = 0.3
        gamma = 0.1
        alpha = 0.1
    elif configuration == config_dense:
        eta = 0.8
        gamma = 1.5
        alpha = 1.5
    else:
        eta = 0.5
        gamma = 1.0
        alpha = 1.0

    Modeller = LssHdpModeller(
            hdp_path=r"..\hdps\hdp",
            input_docs_path=r"..\..\Datasets\pan17_train\problem{}".format(
                    problem_nbr),
            ldac_filename=r"ldac_corpus",
            hdp_output_dir=r"hdp_lss",
            hdp_iters=10000,
            hdp_seed=seed,
            hdp_sample_hyper=False,
            hdp_eta=eta,
            hdp_gamma_s=gamma,
            hdp_alpha_s=alpha,
            word_grams=1,
            drop_uncommon=drop_uncommon,
            freq_threshold=1,
            verbose=verbose)

    # Infer the BoW and LSS representations of the documents
    try:
        # Load, project and visualise the data
        plain_docs, bow_rep_docs, lss_rep_docs = Modeller.get_corpus_lss(
                infer_lss,
                bim=False)

        # Begin Clustering Attempts
        true_labels_path = (r"..\..\Datasets\pan17_train\truth"
                            r"\problem{}\clustering.json"
                            ).format(problem_nbr)

        ground_truth = Tools.load_true_clusters_into_vector(true_labels_path)

        # Normalise the data if not BIM is used!
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

#        ispk_pred, ispk_evals = clu_lss.evaluate(
#                alg_option=Clusterer.alg_iterative_spherical_k_means,
#                param_init="k-means++")

        norm_hdbscan_pred, norm_hdbscan_evals = clu_lss.evaluate(
                alg_option=Clusterer.alg_h_dbscan)

        norm_ms_pred, norm_ms_evals = clu_lss.evaluate(
                alg_option=Clusterer.alg_mean_shift)

#        norm_xm_pred, norm_xm_evals = clu_lss.evaluate(
#                alg_option=Clusterer.alg_x_means)

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

        # Baselines
        bl_rand_pred, bl_rand_evals = clu_lss.evaluate(
                alg_option=Clusterer.bl_random)
        bl_singleton_pred, bl_singleton_evals = clu_lss.evaluate(
                alg_option=Clusterer.bl_singleton)

        nhdp_pred, nhdp_evals = clu_lss.eval_cluster_hdp()
        ntrue_pred, ntrue_evals = clu_lss.eval_true_clustering()

        # SOTA - Gomez et. al. HAC and Log-Entropy with 20k features
        sota_pred_path = (r"D:\College\DKEM\Thesis\AuthorshipClustering\Code"
                          r"\clusterPAN2017-master\train_out_LogEnt"
                          f"\\problem{problem_nbr}\\clustering.json")
        sota_predicted = Tools.load_true_clusters_into_vector(sota_pred_path)
        sota_pred, sota_evals = clu_lss.eval_sota(
                sota_predicted=sota_predicted)

        # Return the results:
        return (Tools.form_problemset_result_dictionary(
                dictionaries=[
                        # ispk_evals, norm_spk_evals, norm_hdbscan_evals,
                        norm_spk_evals, norm_hdbscan_evals,
                        norm_ms_evals,  # norm_xm_evals,
                        nhac_complete_evals, nhac_s_evals, nhac_a_evals,
                        n_optics_evals, bl_rand_evals, bl_singleton_evals,
                        nhdp_evals, sota_evals, ntrue_evals
                        ],
                identifiers=[  # "iSpKmeans",
                             "E_SPKMeans", "E_HDBSCAN",
                             "E_Mean_Shift",  # "XMeans",
                             "E_HAC_C", "E_HAC_Single", "E_HAC_Average",
                             "E_OPTICS", "BL_r", "BL_s",
                             "S_HDP", "BL_SOTA", "Labels"],
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
    problematics = [15, 30, 34, 45, 51, 56, 59, 60]
    scope = range(1, 61)
    print("\n▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬ DROP UNCOMMONS ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\n")

    print("==================== NEUTRAL ====================")
    problemsets_results = []
    k_vals = []
    for ps in scope:
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
            verbose=False,
            configuration=config_neutral,
            drop_uncommon=True)
        problemsets_results.append(ps_result)
        ks = clu.cand_k.copy()
        ks.append(1+max(clu.true_labels))
        k_vals.append(ks)
    my_suffix = "_training_neutral_common"
    info_json = r"..\..\Datasets\pan17_train\info.json"
    Tools.splice_save_problemsets_dictionaries(problemsets_results,
                                               metadata_fpath=info_json,
                                               suffix=my_suffix)
    Tools.save_k_vals_as_df(k_vals=k_vals, suffix=my_suffix)

    print("==================== SPARSE ====================")
    problemsets_results = []
    k_vals = []
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
            seed=13712 * ps,
            infer_lss=False,
            verbose=False,
            configuration=config_sparse,
            drop_uncommon=True)
        problemsets_results.append(ps_result)
        ks = clu.cand_k.copy()
        ks.append(1+max(clu.true_labels))
        k_vals.append(ks)
    my_suffix = "_training_sparse_common"
    info_json = r"..\..\Datasets\pan17_train\info.json"
    Tools.splice_save_problemsets_dictionaries(problemsets_results,
                                               metadata_fpath=info_json,
                                               suffix=my_suffix)
    Tools.save_k_vals_as_df(k_vals=k_vals, suffix=my_suffix)

    print("==================== DENSE ====================")
    problemsets_results = []
    k_vals = []
    for ps in scope:
        print(f"Executing on problem set ► {ps:03d} ◄ ..")
        ps_result, l, lss, plain, clu = problem_set_run(
            problem_set_id=ps,
            n_clusters=None,
            # Emperically specify a random seed that's compatible with
            # hyper sampling and certain problem sets due to a bug in HDP
            # as it seems. However, the seeds would be consistant across
            # runs and yield comparable results for our experiments
            # (comparing different runs of HDP on a problem set)
            seed=None,
            infer_lss=False,
            verbose=False,
            configuration=config_dense,
            drop_uncommon=True)
        problemsets_results.append(ps_result)
        ks = clu.cand_k.copy()
        ks.append(1+max(clu.true_labels))
        k_vals.append(ks)
    my_suffix = "_training_dense_common"
    info_json = r"..\..\Datasets\pan17_train\info.json"
    Tools.splice_save_problemsets_dictionaries(problemsets_results,
                                               metadata_fpath=info_json,
                                               suffix=my_suffix)
    Tools.save_k_vals_as_df(k_vals=k_vals, suffix=my_suffix)
    print("Execution finished.")

    print("\n▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬ KEEP UNCOMMONS ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\n")

    print("==================== NEUTRAL ====================")
    problemsets_results = []
    k_vals = []
    for ps in scope:
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
            verbose=False,
            configuration=config_neutral,
            drop_uncommon=False)
        problemsets_results.append(ps_result)
        ks = clu.cand_k.copy()
        ks.append(1+max(clu.true_labels))
        k_vals.append(ks)
    my_suffix = "_training_neutral_uncommon"
    info_json = r"..\..\Datasets\pan17_train\info.json"
    Tools.splice_save_problemsets_dictionaries(problemsets_results,
                                               metadata_fpath=info_json,
                                               suffix=my_suffix)
    Tools.save_k_vals_as_df(k_vals=k_vals, suffix=my_suffix)

    print("==================== SPARSE ====================")
    problemsets_results = []
    k_vals = []
    for ps in scope:
        print(f"Executing on problem set ► {ps:03d} ◄ ..")
        ps_result, l, lss, plain, clu = problem_set_run(
            problem_set_id=ps,
            n_clusters=None,
            # Emperically specify a random seed that's compatible with
            # hyper sampling and certain problem sets due to a bug in HDP
            # as it seems. However, the seeds would be consistant across
            # runs and yield comparable results for our experiments
            # (comparing different runs of HDP on a problem set)
            seed=13712 * ps,
            infer_lss=False,
            verbose=False,
            configuration=config_sparse,
            drop_uncommon=False)
        problemsets_results.append(ps_result)
        ks = clu.cand_k.copy()
        ks.append(1+max(clu.true_labels))
        k_vals.append(ks)
    my_suffix = "_training_sparse_uncommon"
    info_json = r"..\..\Datasets\pan17_train\info.json"
    Tools.splice_save_problemsets_dictionaries(problemsets_results,
                                               metadata_fpath=info_json,
                                               suffix=my_suffix)
    Tools.save_k_vals_as_df(k_vals=k_vals, suffix=my_suffix)

    print("==================== DENSE ====================")
    problemsets_results = []
    k_vals = []
    for ps in scope:
        print(f"Executing on problem set ► {ps:03d} ◄ ..")
        ps_result, l, lss, plain, clu = problem_set_run(
            problem_set_id=ps,
            n_clusters=None,
            # Emperically specify a random seed that's compatible with
            # hyper sampling and certain problem sets due to a bug in HDP
            # as it seems. However, the seeds would be consistant across
            # runs and yield comparable results for our experiments
            # (comparing different runs of HDP on a problem set)
            seed=None,
            infer_lss=False,
            verbose=False,
            configuration=config_dense,
            drop_uncommon=False)
        problemsets_results.append(ps_result)
        ks = clu.cand_k.copy()
        ks.append(1+max(clu.true_labels))
        k_vals.append(ks)
    my_suffix = "_training_dense_uncommon"
    info_json = r"..\..\Datasets\pan17_train\info.json"
    Tools.splice_save_problemsets_dictionaries(problemsets_results,
                                               metadata_fpath=info_json,
                                               suffix=my_suffix)
    Tools.save_k_vals_as_df(k_vals=k_vals, suffix=my_suffix)
    print("Execution finished.")

    print("▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬ ALL DONE ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬")
