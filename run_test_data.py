# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 21:54:01 2019

@author: RTRAD
"""
from lss_modeller import LssHdpModeller
from aiders import Tools
from clustering import Clusterer
from typing import List, Dict
import warnings

warnings.filterwarnings(action="ignore")  # Supress warning for this code file


class TestApproach:
    # Sampling preference
    config_sparse = "sparse"
    config_neutral = "neutral"
    config_dense = "dense"

    def __init__(self,
                 hdp_exe_path: str,
                 test_corpus_path: str,
                 sampling_iters: int = 10000,
                 sampling_hyper: bool = False,
                 word_grams: int = 1):
        self.hdp_path = hdp_exe_path
        self.test_data_path = test_corpus_path
        self.gibbs_iterations = sampling_iters
        self.gibbs_hyper = sampling_hyper
        self.word_n_grams = word_grams

    def _vectorise_ps(self,
                      ps_id: int,
                      infer_lss: bool,
                      hdp_eta: float,
                      hdp_gamma_s: float,
                      hdp_alpha_s: float,
                      drop_uncommon_terms: bool,
                      frequency_threshold: bool = 1,
                      seed: float = None,
                      bim: bool = False):
        input_ps = f"{self.test_data_path}\\problem{ps_id:03d}"
        lss_modeller = LssHdpModeller(
                hdp_path=self.hdp_path,
                input_docs_path=input_ps,
                ldac_filename=r"ldac_corpus",
                hdp_output_dir=r"lss",
                hdp_iters=self.gibbs_iterations,
                hdp_eta=hdp_eta,
                hdp_gamma_s=hdp_gamma_s,
                hdp_alpha_s=hdp_alpha_s,
                hdp_seed=seed,
                hdp_sample_hyper=self.gibbs_hyper,
                word_grams=self.word_n_grams,
                drop_uncommon=drop_uncommon_terms,
                freq_threshold=frequency_threshold,
                verbose=False)

        return lss_modeller.get_corpus_lss(infer_lss=infer_lss,
                                           bim=bim,
                                           bim_thresold=None)

    def _get_ps_truth(self, ps: int):
        true_labels_path = (r"..\..\Datasets\pan17_test\truth"
                            r"\problem{}\clustering.json"
                            ).format(ps)
        return Tools.load_true_clusters_into_vector(true_labels_path)

    def _cluster_data(self, ps: int, data: List[List]):
        clu_lss = Clusterer(dtm=data,
                            true_labels=ground_truth,
                            max_nbr_clusters=len(lss_rep_docs)-1,
                            min_nbr_clusters=1,
                            min_cluster_size=2,
                            metric="cosine",
                            desired_n_clusters=None)

        norm_spk_pred, norm_spk_evals = clu_lss.evaluate(
                alg_option=Clusterer.alg_spherical_k_means,
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

        k_trend = clu_lss.cand_k
        k_trend.append(1 + max(clu_lss.true_labels))

        result = Tools.form_problemset_result_dictionary(
                dictionaries=[
                        norm_spk_evals, norm_hdbscan_evals,
                        norm_ms_evals, norm_xm_evals,
                        nhac_complete_evals, nhac_s_evals, nhac_a_evals,
                        n_optics_evals,
                        nhdp_evals, ntrue_evals
                        ],
                identifiers=["Spherical_KMeans", "HDBSCAN",
                             "Mean_Shift", "XMeans", "HAC_Complete",
                             "HAC_Single", "HAC_Average", "OPTICS",
                             "HDP", "Labels"],
                problem_set=ps)

        return result, k_trend

    def _save_results(self,
                      suffix: str,
                      info_path: str,
                      results: List[Dict],
                      k_values: List[List]):

        Tools.splice_save_problemsets_dictionaries(
                problemsets_results,
                metadata_fpath=info_path,
                suffix=suffix,
                test_data=True)

        Tools.save_k_vals_as_df(k_vals=k_values, suffix=suffix, test_data=True)

    def run_test(self,
                 configuration: str,
                 drop_uncommon: bool,
                 save_name_suff: str):
        # Adjust the parameters according to the preference
        if configuration == TestApproach.config_sparse:
            eta = 0.3
            gamma = 0.1
            alpha = 0.1
        elif configuration == TestApproach.config_dense:
            eta = 0.8
            gamma = 1.5
            alpha = 1.5
        else:
            eta = 0.5
            gamma = 1.0
            alpha = 1.0

        problemsets_results = []
        k_vals = []
        for ps in range(1, 121):
            print(f"Vectorising problem set ► {ps:03d} ◄ ..")
            plain_docs, bow_rep_docs, lss_rep_docs = tester.vectorise_ps(
                    ps,
                    infer_lss=False,
                    hdp_eta=eta,
                    hdp_gamma_s=gamma,
                    hdp_alpha_s=alpha,
                    drop_uncommon_terms=drop_uncommon)
            lss_rep_docs = Tools.normalise_data(lss_rep_docs)

            # Begin Clustering Attempts
            print("\nClustering ..")
            ps_res, k_trends = tester.cluster_data(ps, lss_rep_docs)
            problemsets_results.append(ps_res)
            k_vals.append(k_trends)

        print("\nSaving Results ..")
        tester.save_results(suffix=save_name_suff,
                            info_path=r"..\..\Datasets\pan17_test\info.json",
                            results=problemsets_results,
                            k_values=k_vals)


if __name__ == "__main__":
    tester = TestApproach(hdp_exe_path=r"..\hdps\hdp",
                          test_corpus_path=r"..\..\Datasets\pan17_test",
                          sampling_iters=10000)

    print("▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\n")
    problemsets_results = []
    k_vals = []
    for ps in range(1, 121):
        print(f"Vectorising problem set ► {ps:03d} ◄ ..")
        plain_docs, bow_rep_docs, lss_rep_docs = tester.vectorise_ps(
                ps,
                infer_lss=False,
                hdp_eta=0.5,
                hdp_gamma_s=1.0,
                hdp_alpha_s=1.0,
                drop_uncommon_terms=True)
        lss_rep_docs = Tools.normalise_data(lss_rep_docs)

        ground_truth = tester.get_ps_truth(ps)

        # Begin Clustering Attempts
        print("\nClustering ..")
        ps_res, k_trends = tester.cluster_data(ps, lss_rep_docs)
        problemsets_results.append(ps_res)
        k_vals.append(k_trends)

    print("\nSaving Results ..")
    tester.save_results(suffix="_normal_drop_1",
                        info_path=r"..\..\Datasets\pan17_test\info.json",
                        results=problemsets_results,
                        k_values=k_vals)

    print("\n▬▬▬▬▬▬▬▬▬▬▬▬(FINISHED)▬▬▬▬▬▬▬▬▬▬▬")
