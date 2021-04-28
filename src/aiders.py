# -*- coding: utf-8 -*-
"""
Aiders and functional tools for the execution of the programme.

"""
import os
from shutil import rmtree
import pandas as pd
import json
from collections import defaultdict
from typing import List, Dict
from sklearn.preprocessing import normalize
from sys import exit
from gensim.matutils import Scipy2Corpus
from gensim.models.logentropy_model import LogEntropyModel
from gensim.matutils import corpus2csc


class Tools:
    """A helper class providing methods for managing files and folders
    and processing data"""

    @staticmethod
    def get_path(path, *paths):
        return os.path.join(path, *paths)

    @staticmethod
    def initialise_directory(dir_path, purge: bool = True):
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
            if purge and os.path.exists(dir_path):
                rmtree(dir_path)
            elif not os.path.exists(dir_path):
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
    def get_lowest_foldername(path) -> str:
        return os.path.basename(os.path.normpath(path))

    @staticmethod
    def split_path(path) -> str:
        return os.path.splitext(path)

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
    def save_list_to_text(mylist: list, filepath: str,
                          header: str = None):
        with open(file=filepath, mode='w', encoding="utf8") as file_handler:
            if header:
                file_handler.write(f"{header}\n{'-'*12}\n")
            for item in mylist:
                file_handler.write(f"{item}\n")

    @staticmethod
    def normalise_data(data: List[List],
                       log_e: bool,
                       normalise_log_e: bool = False):
        # Form a normalised dataframe with the same index
        idx = data.index
        if log_e:
            lss_c = Scipy2Corpus(data.values)
            model = LogEntropyModel(lss_c, normalize=normalise_log_e)
            # Convert gensims transformed corpus to array corpus for clustering
            lss = corpus2csc(model[lss_c]).T.toarray()
            return pd.DataFrame(data=lss,
                                index=idx)
        else:
            return pd.DataFrame(data=normalize(data, norm="l2"),
                                index=idx)

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
    def calc_rmse(x: pd.Series,
                  y: pd.Series):
        return ((x - y) ** 2).mean() ** .5

    @staticmethod
    def analyse_true_k(truth_path: str,
                       ps_range: range):
        k = []
        for ps in ps_range:
            fp = os.path.join(truth_path, f"problem{ps:03d}",
                              "clustering.json")
            k.append(len(Tools._read_json_file(fp)))
        return pd.Series(k).describe()

    @staticmethod
    def get_sota_est_k(output_path: str):
        vals = []
        for ps in range(1, 121):
            fp = os.path.join(output_path, f"problem{ps:03d}",
                              "clustering.json")
            vals.append(len(Tools._read_json_file(fp)))
        return pd.Series(vals)

    @staticmethod
    def analyse_LSSR_times(m_tr_path: str,
                           m_te_path: str):
        times = []
        # Consume training times
        for ps in range(1, 61):
            fp = os.path.join(
                m_tr_path, f"problem{ps:03d}",
                "hdp_lss_0.30_0.10_0.10_common_True", "state.log")
            df = pd.read_csv(fp, delim_whitespace=True, usecols=["time"])
            times.append(df.iloc[-1, 0])
        # Consume testing times
        for ps in range(1, 121):
            fp = os.path.join(
                m_te_path, f"problem{ps:03d}",
                "lss_0.30_0.10_0.10_common_True", "state.log")
            df = pd.read_csv(fp, delim_whitespace=True, usecols=["time"])
            times.append(df.iloc[-1, 0])

        return pd.Series(times).describe()


def main():
    print("Aiders here")
    exit(0)


if __name__ == "__main__":
    main()
