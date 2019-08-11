# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 21:54:01 2019

@author: RTRAD
"""
from lss_modeller import LssHdpModeller
from aiders import Tools
from clustering import Clusterer
import warnings
warnings.filterwarnings(action="ignore")  # Supress warning for this code file


class TestApproach:

    def __init__(self,
                 hdp_exe_path: str,
                 test_corpus_path: str,
                 sampling_iters: int = 10000,
                 sampling_hyper: bool = False,
                 sampling_eta: float = 0.5,
                 word_grams: int = 1):
        self.hdp_path = hdp_exe_path
        self.test_data_path = test_corpus_path
        self.gibbs_iterations = sampling_iters
        self.gibbs_hyper = sampling_hyper
        self.word_n_grams = word_grams

    def vectorise_ps(self,
                     ps_id: int,
                     infer_lss: bool = False,
                     seed: float = 13712):
        input_ps = f"{self.test_data_path}\\problem{ps_id:03d}"
        lss_modeller = LssHdpModeller(hdp_path=self.hdp_path,
                                      input_docs_path=input_ps,
                                      ldac_filename=r"ldac_corpus",
                                      hdp_output_dir=r"lss",
                                      hdp_iters=self.gibbs_iterations,
                                      hdp_seed=seed,
                                      hdp_sample_hyper=self.gibbs_hyper,
                                      word_grams=self.word_n_grams,
                                      verbose=False)
        return lss_modeller.get_corpus_lss(infer_lss=infer_lss)


if __name__ == "__main__":
    tester = TestApproach(hdp_exe_path=r"..\hdps\hdp",
                          test_corpus_path=r"..\..\Datasets\pan17_test")

    print("▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\n")
    for ps in range(1, 121):
        print(f"Vectorising problem set ► {ps:03d} ◄ ..")
        plain_docs, bow_rep_docs, lss_rep_docs = tester.vectorise_ps(
                ps,
                infer_lss=True)
    print("▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\n")