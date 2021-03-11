# -*- coding: utf-8 -*-
"""
Cluster a set of documents whose LSSR has been already built.

@author: trad
"""

from root_logger import logger
import argparse
from aiders import Tools
from clustering import Clusterer
from typing import List, Dict
from collections import defaultdict
import pandas as pd
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

    path = r"{}\mode-word-assignments.dat".format(
            lssr_dirpath)

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
                    name, ext = Tools.split_path(doc.path)
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
                 out_dir: str, my_suffix: str, my_index: list):
    integrated_results = defaultdict(list)
    for r in results:
        if r is None:
            continue
        for k in r.keys():
            integrated_results[k].append(r[k])

    df = pd.DataFrame(data=integrated_results)
    df.index = my_index

    timestamp = pd.to_datetime("now").strftime("%Y%m%d_%H%M%S")
    df.to_csv(
        path_or_buf=(f"{out_dir}\\{timestamp}_authorial_clustering_results"
                     f"_{my_suffix}.csv"),
        index=True)


def main():

    parser = argparse.ArgumentParser(
        description="Cluster a set of documents relying on their LSSR",
        epilog=("Please refer to the README page of the repository "
                "and the requirement.txt file in case problems occur."))

    # Specify the arguments
    parser.add_argument(
        "input_docs_folderpath"
        )
    parser.add_argument(
        "lssr_dir",
        help=("The LSSR which resulted from HDP. "
              "If still not built, you can build it using the other script: "
              "lssr_doc, which calls hdp.exe implicitly."))
    parser.add_argument(
        "ground_truth",
        help=("The JSON ground truth file of the clustering problem, "
              "like PAN-17 format."))
    parser.add_argument(
        "out_dir",
        help="The output directory to write the final results to.")
    parser.add_argument(
        "-k", "--desired_n_clusters",  type=int, default=None,
        help=("The desired k, number of clusters. "
              "By default k will be automatically selected, "
              "but you can enter 0 to use the true k."))
    parser.add_argument("-raw", "--use_raw_counts", action="store_true")
    parser.add_argument("-l_percent", "--ml_cl_constraints_percentage",
                        type=float, default=12)
    parser.add_argument("-suffix", "--results_fname_suffix",
                        type=str, default="")
    parser.add_argument("-v", "--verbose", action="store_true")
    # Parse arguments from sys.args
    args = parser.parse_args()

    # Load the ground truth for experimentation
    ground_truth = Tools.load_true_clusters_into_vector(args.ground_truth)

    # Load and normalise lSSR
    lssr = load_lss_representation_into_df(
        lssr_dirpath=args.lssr_dir,
        input_docs_folderpath=args.input_docs_folderpath,
        normalise=not args.use_raw_counts)
    logger.info("LSSR loaded successfully")

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

    idx = []
    res = []
    kvals = []

    # Baselines
    bl_rand_pred, bl_rand_evals = clu_lss.evaluate(
            alg_option=Clusterer.bl_random)
    idx.append("BL_Random")
    res.append(bl_rand_evals)
    kvals.append(bl_rand_pred)

    bl_singleton_pred, bl_singleton_evals = clu_lss.evaluate(
            alg_option=Clusterer.bl_singleton)
    idx.append("BL_Singleton")
    res.append(bl_singleton_evals)
    kvals.append(bl_singleton_pred)

    # Clustering algorithms
    norm_spk_pred, norm_spk_evals = clu_lss.evaluate(
            alg_option=Clusterer.alg_spherical_k_means,
            param_init="k-means++")
    idx.append("SPKMeans")
    res.append(norm_spk_evals)
    kvals.append(norm_spk_pred)

    logger.info("Spherical KMeans clustering done")

    cop_kmeans_pred, cop_kmeans_evals = clu_lss.evaluate(
        alg_option=Clusterer.alg_cop_kmeans,
        param_constraints_size=args.ml_cl_constraints_percentage/100,
        param_copkmeans_init="random")
    idx.append("COP_KMeans")
    res.append(cop_kmeans_evals)
    kvals.append(cop_kmeans_pred)

    logger.info("Constrained KMeans clustering done")

    # Saving results:
    save_results(results=res, k_pred=kvals,
                 out_dir=args.out_dir, my_suffix=args.results_fname_suffix,
                 my_index=idx)

    logger.info(f"Execution completed and results saved under {args.out_dir}.")
    logger.shutdown()


if __name__ == "__main__":
    main()
