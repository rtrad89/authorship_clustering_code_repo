# -*- coding: utf-8 -*-
"""
Vectorise a set of documents and represent them as vectors in a vector space
model using HDP non-parametric topic model.

@author: trad
"""

from src.root_logger import logger
import argparse
from src.lss_modeller import LssHdpModeller


def main():

    parser = argparse.ArgumentParser(
        description="Vectorise and build the LSSR of a set of documents",
        epilog=("Please refer to the README page of the repository "
                "and the requirement.txt file in case problems occur."))
    # Specify the arguments
    parser.add_argument("input_docs_folderpath")
    parser.add_argument("hdp_exe_path")
    parser.add_argument("-iter", "--gibbs_iters",  type=int, default=1000)
    parser.add_argument("-hyper", "--gibbs_hypersampling", action="store_true")
    parser.add_argument("-e", "--eta", type=float, default=0.3)
    parser.add_argument("-g", "--gamma", type=float, default=0.1)
    parser.add_argument("-a", "--alpha", type=float, default=0.1)
    parser.add_argument("-ngrams", "--word_ngrams", type=int, default=1)
    parser.add_argument("-drop", "--drop_uncommons", action="store_true")
    parser.add_argument("-dropf", "--drop_frequency_threshold", type=int,
                        default=1)
    parser.add_argument("-v", "--verbose", action="store_true")
    # Parse arguments from sys.args
    args = parser.parse_args()

    # Initialise and run the LSSR modeller#
    lssr_modeller = LssHdpModeller(
        hdp_path=args.hdp_exe_path,
        input_docs_path=args.input_docs_folderpath,
        ldac_filename=r"ldac_corpus",
        hdp_output_dir=r"hdp_lss",
        hdp_iters=args.gibbs_iters,
        hdp_eta=args.eta,
        hdp_gamma_s=args.gamma,
        hdp_alpha_s=args.alpha,
        hdp_seed=13712,
        hdp_sample_hyper=args.gibbs_hypersampling,
        word_grams=args.word_ngrams,
        drop_uncommon=args.drop_uncommons,
        freq_threshold=args.drop_frequency_threshold,
        verbose=args.verbose)

    _, _, _ = lssr_modeller.get_corpus_lss(infer_lss=True)
    logger.shutdown()


if __name__ == "__main__":
    main()
