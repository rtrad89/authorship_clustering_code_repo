# -*- coding: utf-8 -*-
"""
Cluster a set of documents whose LSSR has been already built.

@author: trad
"""

from src.root_logger import logger
import argparse
from src.aiders import Tools
from src.clustering import Clusterer
from typing import List, Dict
from collections import defaultdict
import pandas as pd
from numpy import unique
from sys import exit as sysexit
import warnings

warnings.filterwarnings(action="ignore")  # Supress warning for this code file


def load_lss_representation_into_df(lssr_dirpath,
                                    input_docs_folderpath,
                                    normalise: bool = True):
    """
    Load and normalise BoT LSSR from disk to a returned dataframe.

    Returns
    -------
    lss_df : pd.DataFrame
        A matrix of shape (n_samples, n_features)

    Raises
    ------
    FileNotFoundError
        When the LSSR isn't found on disk.

    """

    path = Tools.get_path(lssr_dirpath, "mode-word-assignments.dat")

    # We need word counts under each topic, to produce some sort
    # of a bag-of-topics model (BoT)
    try:
        lss_df = pd.read_csv(filepath_or_buffer=path,
                             delim_whitespace=True)
        lss_df = lss_df.pivot_table(
                values='w', columns='z', index='d',
                aggfunc='count', fill_value=0)

        # Index with file names for later reference
        if lss_df.index.is_numeric():
            doc_index = []
            # We will need to build the index
            with Tools.scan_directory(input_docs_folderpath) as docs:
                for doc in docs:
                    _, ext = Tools.split_path(doc.path)
                    # Add the file to index if it's a text file only
                    if ext == ".txt":
                        doc_index.append(Tools.get_filename(doc.path))
            lss_df.index = doc_index

        if normalise:
            lss_df = Tools.normalise_data(lss_df, log_e=False)
        return lss_df

    except FileNotFoundError:
        logger.error(("\nNo LSS precomputed file was found on disk via:\n{}\n"
                      "> Please generate LDA-C corpus and run HDP first...\n"
                      ).format(path))


def save_results(results: List[Dict], k_pred: List[List],
                 out_dir: str, my_suffix: str, my_index: list, n_corpora: int,
                 mode: chr, corpus_names: List[str]):

    integrated_results = defaultdict(list)

    if not mode == "m":
        for r in results:
            if r is None:
                continue
            for k in r.keys():
                integrated_results[k].append(r[k])
    else:
        for r in results:
            if r is None:
                continue
            for k in r.keys():
                integrated_results[k].extend([r[k]])

    df_res = pd.DataFrame(data=integrated_results)
    # Make a multi-index of corpus name and method name combined:
    df_res.index = [[name for name in corpus_names
                     for i in range(len(my_index))],
                    my_index*n_corpora]

    if not mode == "m":
        # Convert all internal elements to lists and then make k_vals dataframe
        df_k_vals = pd.DataFrame(k_pred, index=my_index,
                                 columns=["k_estimations"]).T
    else:
        df_k_vals = pd.DataFrame(k_pred, columns=my_index)

    timestamp = pd.to_datetime("now").strftime("%Y%m%d_%H%M%S")

    Tools.initialise_directories(out_dir)

    # Construct the results path
    save_path = Tools.get_path(
        out_dir, f"{timestamp}_authorial_clustering_results_{my_suffix}.csv")
    df_res.to_csv(
        path_or_buf=save_path,
        index=True)

    save_path = Tools.get_path(
        out_dir, f"{timestamp}_authorial_clustering_kvals_{my_suffix}.csv")
    df_k_vals.to_csv(
        path_or_buf=save_path,
        index=True)


def single_run(args):
    # Load the ground truth for experimentation
    ground_truth = Tools.load_true_clusters_into_vector(
        Tools.get_path(
            args.ground_truth,
            Tools.get_lowest_foldername(args.input_docs_folderpath),
            "clustering.json")
        )

    # Load and normalise lSSR
    lssr = load_lss_representation_into_df(
        lssr_dirpath=Tools.get_path(
            args.input_docs_folderpath, args.lssr_dir_name),
        input_docs_folderpath=args.input_docs_folderpath,
        normalise=not args.use_raw_counts)

    if not lssr.empty:
        logger.info("LSSR loaded successfully")
    else:
        logger.info("LSSR couldn't be loaded "
                    "(have you run HDP first "
                    "and used the correct lssr_dir_name?)")
        sysexit(-1)

    if args.verbose:
        logger.info("LSSR:", lssr, "\n")

    # Initialise and run the clusterer module
    clu_lss = Clusterer(dtm=lssr,
                        true_labels=ground_truth,
                        max_nbr_clusters=len(lssr)-1,
                        min_nbr_clusters=1,
                        min_cluster_size=2,
                        metric="cosine",
                        desired_n_clusters=args.desired_n_clusters)

    idx, res, kvals = [], [], []

    # Baselines
    bl_rand_pred, bl_rand_evals = clu_lss.evaluate(
            alg_option=Clusterer.bl_random)
    idx.append("BL_Random")
    res.append(bl_rand_evals)
    kvals.append(len(unique(bl_rand_pred)))

    bl_singleton_pred, bl_singleton_evals = clu_lss.evaluate(
            alg_option=Clusterer.bl_singleton)
    idx.append("BL_Singleton")
    res.append(bl_singleton_evals)
    kvals.append(len(unique(bl_singleton_pred)))

    ntrue_pred, ntrue_evals = clu_lss.eval_true_clustering()
    idx.append("Ground_Truth")
    res.append(ntrue_evals)
    kvals.append(len(unique(ntrue_pred)))

    # Clustering algorithms
    norm_spk_pred, norm_spk_evals = clu_lss.evaluate(
            alg_option=Clusterer.alg_spherical_k_means,
            param_init="k-means++")
    idx.append("SPKMeans")
    res.append(norm_spk_evals)
    kvals.append(len(unique(norm_spk_pred)))

    logger.info("Spherical KMeans clustering done")

    cop_kmeans_pred, cop_kmeans_evals = clu_lss.evaluate(
        alg_option=Clusterer.alg_cop_kmeans,
        param_constraints_size=args.ml_cl_constraints_percentage/100,
        param_copkmeans_init="random")
    idx.append("COP_KMeans")
    res.append(cop_kmeans_evals)
    kvals.append(len(unique(cop_kmeans_pred)))

    fail_flag = all(val is None for val in cop_kmeans_evals.values())
    logger.info("Constrained KMeans failed to satisfy all constraints at the "
                "prespecified k. This can happen when an unfit k value is "
                "entered by users manually using -k option." if fail_flag
                else "Constrained KMeans clustering done.")

    return res, kvals, idx


def main():

    parser = argparse.ArgumentParser(
        description="Cluster a set of documents relying on their LSSR",
        epilog=("Please refer to the README page of the repository\n"
                "https://github.com/rtrad89/authorship_clustering_code_repo\n"
                "and the requirement.txt file in case problems occur."))

    # Specify the arguments
    parser.add_argument(
        "operation_mode",
        help=("Operation mode of the code: s for a single corpus, where a "
              "folder of text documents is expected; "
              "m for multiple corpora where a directory of folders of text "
              "files is expected as input, each folder representing a corpus.")
        )
    parser.add_argument(
        "input_docs_folderpath",
        help="The directory of the corpus/corpora."
        )
    parser.add_argument(
        "lssr_dir_name",
        help=("The LSSR folder name which resulted from HDP. "
              "These are expected to be inside each corpus folder."))
    parser.add_argument(
        "ground_truth",
        help=("The ground truth folder of the clustering problem(s), "
              "where there is a folder for each corpus and named identically, "
              "containing clustering.json files, like PAN-17 dataset."))
    parser.add_argument(
        "output_dir",
        help="The directory where the outputs shall be saved."
        )
    parser.add_argument(
        "-k", "--desired_n_clusters",  type=int, default=None,
        help=("The desired k, number of clusters. "
              "By default, k will be automatically selected, "
              "but you can enter a value of your choice, "
              "or 0 to try use the true k if possible."))
    parser.add_argument(
        "-raw", "--use_raw_counts", action="store_true",
        help=("By default, L2 normalisation will be applied "
              "to the term frequencies. Specify this "
              "option to use raw counts instead."))
    parser.add_argument(
        "-l_percent", "--ml_cl_constraints_percentage",
        type=float, default=12,
        help=("Specify the ML/CL constraint coverage. "
              "By default it is 12. For more details refer "
              "to the paper."))
    parser.add_argument(
        "-suffix", "--results_fname_suffix",
        type=str, default="",
        help=("A suffix for the name of the results file, "
              "if desired."))
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Specify for verbose outputs.")
    # Parse arguments from sys.args
    args = parser.parse_args()

    # assemble the output directory
    out_dir = args.output_dir

    # Execute single run
    if args.operation_mode != "m":
        res, kvals, idx = single_run(args)
        my_n_corpora = 1
        my_corpus_names = [Tools.get_lowest_foldername(
            args.input_docs_folderpath)]
    else:
        my_corpus_names, res, kvals, idx = [], [], [], []
        my_n_corpora = 0
        with Tools.scan_directory(args.input_docs_folderpath) as dirs:
            for folder in dirs:
                if not Tools.is_path_dir(folder):
                    continue
                args.input_docs_folderpath = folder.path
                try:
                    corpus = Tools.get_lowest_foldername(folder.path)
                    logger.info(f"> Clustering \"{corpus}\"")
                    single_res, single_kvals, idx = single_run(
                        args)
                    my_corpus_names.append(corpus)
                    res.extend(single_res)
                    kvals.append(single_kvals)
                    my_n_corpora += 1
                except Exception as e:
                    logger.error(f"Clustering failed with the message:\n"
                                 f"{str(e)}")
                    logger.info(f"\t skipping {folder.path}")
                    continue

    # Check if all clustering problems proceeded as desired
    assert len(res) == my_n_corpora*len(idx), "Some corpora clustering failed!"

    # Saving results:
    save_results(results=res, k_pred=kvals,
                 out_dir=out_dir, my_suffix=args.results_fname_suffix,
                 my_index=idx, n_corpora=my_n_corpora,
                 mode=args.operation_mode, corpus_names=my_corpus_names)

    logger.info(f"Execution completed and results saved under {out_dir}.")
    logger.shutdown()


if __name__ == "__main__":
    main()
