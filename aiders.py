# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:27:46 2019

@author: RTRAD
"""
import os
from shutil import rmtree
import pandas as pd
import json
from collections import defaultdict
from typing import List, Dict
import powerlaw
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman
from sys import exit


class Tools:
    """A helper class providing methods for managing files and folders
    and processing data"""

    @staticmethod
    def initialise_directory(dir_path):
        """
        Ensure an empty directory is created in `dir_path`.

        Parameters
        ----------
        dir_path : str
            The path of the desired directory.

        Raises
        ------
        PermissionError
            If `dir_path` is not accessible by the current user.
        """

        try:
            if os.path.exists(dir_path):
                rmtree(dir_path)
            os.mkdir(dir_path)
        except PermissionError:
            print("ERROR: Please make sure the folders required by the program"
                  "are not already opened")

    @staticmethod
    def remove_directory(dir_path):
        if os.path.exists(dir_path):
            rmtree(dir_path)

    @staticmethod
    def initialise_directories(dir_path):
        """
        Ensure an empty directory is created in `dir_path`, guaranteeing that
        all the needed directories on the path are also created

        Parameters
        ----------
        dir_path : str
            The path of the desired directory.

        Raises
        ------
        PermissionError
            If `dir_path` is not accessible by the current user.
        """

        try:
            if os.path.exists(dir_path):
                rmtree(dir_path)
            os.makedirs(dir_path)
        except PermissionError:
            print("ERROR: Please make sure the folders required by the program"
                  "are not already opened")

    @staticmethod
    def get_filename(path) -> str:
        return os.path.basename(path)

    @staticmethod
    def _read_json_file(path) -> object:
        with open(path, mode='r', encoding='utf8') as json_file:
            json_data = json_file.read()
        return json.loads(json_data)

    @staticmethod
    def load_true_clusters_into_df(path) -> pd.DataFrame:
        # Read json file contents
        json_content = Tools._read_json_file(path=path)

        # The data is read as a list of list of dictionaries
        # Transofrming it to an indexed series with labels for compatability
        ground_truth = {}
        for idx in range(0, len(json_content)):
            ground_truth.update(
                    {idx: [list(x.values()).pop() for x in json_content[idx]]}
                    )

        return pd.DataFrame(data=[ground_truth],
                            index=["clusters"]).transpose()

    @staticmethod
    def scan_directory(path):
        return os.scandir(path)

    @staticmethod
    def path_exists(path):
        return os.path.exists(path)

    @staticmethod
    def is_path_dir(path):
        return os.path.isdir(path)

    @staticmethod
    def load_true_clusters_into_vector(path,
                                       sort: bool = False) -> pd.Series:
        """
        Load the true clustering json file into a series indexed by file names.

        Returns
        -------
        vec : pd.Series
            Cluster labels of the documents in a series whose index is the
            names of the files and values are the lables.

        """
        # Read the json file
        json_contents = Tools._read_json_file(path=path)
        # Dismantle the nested dictionaries
        temp_list = []
        for idx in range(0, len(json_contents)):
            temp_list.append(
                    [list(clus.values()).pop() for clus in json_contents[idx]])
        # Reshape the list as a series indexed with file names
        temp_dict = {}
        for idx in range(0, len(temp_list)):
            for name in temp_list[idx]:
                temp_dict.update({name: idx})

        vec = pd.Series(temp_dict, name="true")
        return (vec if not sort else vec.sort_index())

    @staticmethod
    def form_problemset_result_dictionary(dictionaries: List[Dict],
                                          identifiers: List[str],
                                          problem_set: int):
        res = defaultdict(list)
        res["set"].extend([f"problem{problem_set:03d}"] * len(dictionaries))
        for i, d in enumerate(dictionaries):
            res["algorithm"].append(identifiers[i])
            for k in d.keys():
                res[k].append(d[k])
        return res

    @staticmethod
    def splice_save_problemsets_dictionaries(ps_dicts: List[Dict],
                                             metadata_fpath: str,
                                             suffix: str = "",
                                             test_data: bool = False):
        """
        Parameters
        ----------
        metadata_fpath : str
            The path to the info.json file which accompanys the clustering.
        """
        integrated_results = defaultdict(list)
        for r in ps_dicts:
            if r is None:
                continue
            for k in r.keys():
                integrated_results[k].extend(r[k])

        df = pd.DataFrame(data=integrated_results)
        # Merge them with metadata about the clusterings and remove last
        # redundant column as well
        metadata = pd.DataFrame(Tools._read_json_file(metadata_fpath))
        df = df.merge(metadata,
                      left_on="set",
                      right_on="folder").iloc[:, 0:-1]

        if len(df) > 0:
            timestamp = pd.to_datetime("now").strftime("%Y%m%d_%H%M%S")
            if test_data:
                path = f"./__outputs__/TESTS/results_{timestamp}{suffix}.csv"
            else:
                path = f"./__outputs__/results_{timestamp}{suffix}.csv"
            df.sort_values(by=["set", "bcubed_fscore"], ascending=[True, False]
                           ).to_csv(path_or_buf=path,
                                    index=False)
        return path

    @staticmethod
    def test_power_law_dist(true_labels_dir: str):
        data = []
        for ps in range(1, 61):
            problem_nbr = f"{ps:03d}"
            path = r"{}/problem{}/clustering.json".format(
                    true_labels_dir,
                    problem_nbr)
            temp = Tools.load_true_clusters_into_vector(path)
            data.extend(list(temp.value_counts()))

        # Show the data:
        plt.figure(dpi=600)
        s = pd.Series(data)
        s.value_counts().sort_index().plot.bar(x="cluster size")
        plt.xlabel(xlabel="Size of cluster")
        plt.ylabel(ylabel="Number of clusters")
        plt.grid(b=True, axis="y")
        plt.gcf().savefig("./data_hist.pdf")
        plt.close()
        # Fit a power law distribution to the cluster sizes data:
        fit = powerlaw.Fit(data=data, estimate_discrete=True)
        print("▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬")
        print(f"Fit was finished with:"
              f"\n\t→ alpha={fit.alpha:0.3f}"
              f"\n\t→ sigma={fit.sigma:0.3f}"
              f"\n\t→ x_min={fit.xmin:0.3f}")
        # Compate the fit to other distributions:
        candidates = ["truncated_power_law",
                      "lognormal",
                      "lognormal_positive",
                      "exponential"]
        print("---------------------------------")
        for val in candidates:
            R, p = fit.distribution_compare("power_law", val)
            print(f"powerlaw <> {val}: R={R:0.3f}, p={p:0.3f}")
        print("▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬")

        return s, fit

    @staticmethod
    def save_list_to_text(mylist: list, filepath: str,
                          header: str = None):
        with open(filepath, 'w') as file_handler:
            if header:
                file_handler.write(f"{header}\n{'-'*12}\n")
            for item in mylist:
                file_handler.write(f"{item}\n")

    @staticmethod
    def normalise_data(data: List[List]):
        # Form a normalised dataframe with the same index
        return pd.DataFrame(data=normalize(data, norm="l2"),
                            index=data.index)

    @staticmethod
    def save_k_vals_as_df(k_vals: List[List],
                          cop_kmeans_frac: float,
                          suffix="",
                          test_data: bool = False):
        df_k_vals = pd.DataFrame(k_vals,
                                 columns=["E_SPKMeans",
                                          "Gap",
                                          "G-means",
                                          "E_COP_KMeans",
                                          "E_HAC_C",
                                          "E_HAC_S",
                                          "E_HAC_A",
                                          "E_OPTICS",
                                          "TRUE"])

        timestamp = pd.to_datetime("now").strftime("%Y%m%d_%H%M%S")
        if test_data:
            path = f"./__outputs__/TESTS/k_trend_{timestamp}{suffix}.csv"
        else:
            path = f"./__outputs__/k_trend_{timestamp}{suffix}.csv"

        df_k_vals.to_csv(path)

    @staticmethod
    def friedman_nemenyi_bonferroni_tests(
            data_path: str, ari_included=False, save_outputs: bool = False):
        # Load the data from disk
        df = pd.read_csv(data_path, low_memory=False)

        if ari_included:
            alpha = .05 / 2.0  # Bonferroni Correction with 2 tests
        else:
            alpha = .05

        # Reshape the results so that the treatments (algorithms) are
        # arranged in a columnar fashion
        pvt_b3f = df.pivot_table(
                values="bcubed_fscore", columns="algorithm",
                index=["language", "genre"])
        pvt_b3f.index = pvt_b3f.index.map("_".join)

        pvt_b3f.drop(
                columns=["E_HAC_Average", "E_HAC_Single", "E_HDBSCAN",
                         "Labels"],
                inplace=True)

        stat_b3f, p_b3f = friedmanchisquare(*pvt_b3f.T.values)
        sig_b3f = p_b3f <= alpha

        posthoc_b3f = None
        if sig_b3f:
            # The omnibus test succeeded, drilling down to the posthoc test
            posthoc_b3f = posthoc_nemenyi_friedman(pvt_b3f)
            if save_outputs:
                posthoc_b3f.to_csv(
                        r"./__outputs__/TESTS"
                        f"/Friedman_Nemenyi_B3F_a_{alpha:0.4f}.csv")

        if ari_included:
            pvt_ari = df.pivot_table(
                    values="ari", columns="algorithm",
                    index=["language", "genre"])
            pvt_ari.index = pvt_ari.index.map("_".join)

            pvt_ari.drop(
                    columns=["E_HAC_Average", "E_HAC_Single", "E_HDBSCAN",
                             "Labels"],
                    inplace=True)
            stat_ari, p_ari = friedmanchisquare(*pvt_ari.T.values)
            sig_ari = p_ari <= alpha
            posthoc_ari = None
            if sig_ari:
                # The omnibus test succeeded, drilling down to the posthoc test
                posthoc_ari = posthoc_nemenyi_friedman(pvt_ari)
                if save_outputs:
                    posthoc_ari.to_csv(
                            r"./__outputs__/TESTS/"
                            f"Friedman_Nemenyi_ARI_a_{alpha:0.4f}.csv")

        if ari_included:
            ret = alpha, sig_b3f, posthoc_b3f, sig_ari, posthoc_ari
        else:
            ret = alpha, sig_b3f, posthoc_b3f

        return ret

    @staticmethod
    def calc_rmse(x: pd.Series,
                  y: pd.Series):
        return ((x - y) ** 2).mean() ** .5

    @staticmethod
    def analyse_true_k(truth_path: str,
                       ps_range: range):
        k = []
        for ps in ps_range:
            fp = f"{truth_path}\\problem{ps:03d}\\clustering.json"
            k.append(len(Tools._read_json_file(fp)))
        return pd.Series(k).describe()

    @staticmethod
    def get_sota_est_k(output_path: str):
        vals = []
        for ps in range(1, 121):
            fp = f"{output_path}\\problem{ps:03d}\\clustering.json"
            vals.append(len(Tools._read_json_file(fp)))
        return pd.Series(vals)

    @staticmethod
    def analyse_LSSR_times(m_tr_path: str,
                           m_te_path: str):
        times = []
        # Consume training times
        for ps in range(1, 61):
            fp = (f"{m_tr_path}\\problem{ps:03d}"
                  "\\hdp_lss_0.30_0.10_0.10_common_True\\state.log")
            df = pd.read_csv(fp, delim_whitespace=True, usecols=["time"])
            times.append(df.iloc[-1, 0])
        # Consume testing times
        for ps in range(1, 121):
            fp = (f"{m_te_path}\\problem{ps:03d}"
                  "\\lss_0.30_0.10_0.10_common_True\\state.log")
            df = pd.read_csv(fp, delim_whitespace=True, usecols=["time"])
            times.append(df.iloc[-1, 0])

        return pd.Series(times).describe()


def main():
    print("Aiders here")
    exit(0)


if __name__ == "__main__":
    main()
