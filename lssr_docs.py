# -*- coding: utf-8 -*-
"""
Vectorise a set of documents and represent them as vectors in a vector space
model using HDP non-parametric topic model.

@author: trad
"""

from src.root_logger import logger
import argparse
from src.lss_modeller import LssHdpModeller
from src.aiders import Tools
# from sys import exit as sysexit
import warnings

warnings.filterwarnings(action="ignore")  # Supress warning for this code file


def single_run(args):
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

    lssr_modeller.get_corpus_lss(infer_lss=True)


def main():

    parser = argparse.ArgumentParser(
        description="Vectorise and build the LSSR of a set of documents",
        epilog=("Please refer to the README page of the repository\n"
                "https://github.com/rtrad89/authorship_clustering_code_repo\n"
                "and the requirement.txt file in case problems occur."))
    # Specify the arguments
    parser.add_argument(
        "operation_mode",
        help=("Operation mode of the code: s for a single corpus, "
              "where a folder of text documents is expected; m for "
              "multiple corpora where a directory of folders of text "
              "files is expected as input, each folder representing a corpus.")
        )
    parser.add_argument(
        "input_docs_folderpath",
        help=("The path of the corpus/corpora folder."))
    parser.add_argument(
        "hdp_exe_path",
        help=("The path of hdp.exe programme, which can be compiled using the "
              "code on https://github.com/blei-lab/hdp"))
    parser.add_argument(
        "-iter", "--gibbs_iters",  type=int, default=10000,
        help=("The number of Gibbs sampler's iterations. Default = 10000."))
    parser.add_argument(
        "-hyper", "--gibbs_hypersampling", action="store_true",
        help=("Specify to use hyper-sampling."))
    parser.add_argument(
        "-e", "--eta", type=float, default=0.3,
        help="eta sampling parameter "
        "(please see https://doi.org/10.1198/016214506000000302 for details).")
    parser.add_argument(
        "-g", "--gamma", type=float, default=0.1,
        help="gamma concentration parameter "
        "(please see https://doi.org/10.1198/016214506000000302 for details).")
    parser.add_argument(
        "-a", "--alpha", type=float, default=0.1,
        help="alpha concentration parameter "
        "(please see https://doi.org/10.1198/016214506000000302 for details).")
    parser.add_argument(
        "-ngrams", "--word_ngrams", type=int, default=1,
        help="Vectorisation word grams value, which is 1 by default.")
    parser.add_argument(
        "-drop", "--drop_uncommons", action="store_true",
        help=("Specify to drop words which occur infrequently in a corpus. "
              "frequency threshold is 1 by default, and can be changed using "
              "--drop_frequency_threshold parameter.")
        )
    parser.add_argument(
        "-dropf", "--drop_frequency_threshold", type=int, default=1,
        help=("The frequency threshold whereby words are deemed uncommon "
              "for dropping if --drop_uncommons is specified."))
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Specify for verbose outputs.")
    # Parse arguments from sys.args
    args = parser.parse_args()

    # Depending on operation mode, either run a single iteration or multiple
    if args.operation_mode != "m":
        single_run(args)
    else:
        copy_args = args
        with Tools.scan_directory(args.input_docs_folderpath) as dirs:
            for folder in dirs:
                if not Tools.is_path_dir(folder):
                    continue
                copy_args.input_docs_folderpath = folder.path
                logger.info(f"\n Processing \"{folder.path}\" \n")
                try:
                    single_run(copy_args)
                except Exception as e:
                    logger.error(f"Processing failed with the message:\n"
                                 f"{str(e)}")
                    logger.info(f"\t skipping {folder.path}")
                    continue

    logger.shutdown()


if __name__ == "__main__":
    main()
