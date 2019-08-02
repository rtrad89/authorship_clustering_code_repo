# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:27:46 2019

@author: RTRAD
"""
import os
from shutil import rmtree
import pandas as pd
import gzip
import json
from collections import defaultdict
from typing import List, Dict
import powerlaw
import matplotlib.pyplot as plt


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
    def load_true_clusters_into_vector(path) -> pd.Series:
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
        return vec

    @staticmethod
    def splice_save_problemsets_dictionaries(ps_dicts: List[Dict]):
        integrated_results = defaultdict(list)
        for r in ps_dicts:
            if r is None:
                continue
            for k in r.keys():
                integrated_results[k].extend(r[k])

        df = pd.DataFrame(data=integrated_results)

        if len(df) > 0:
            timestamp = str(pd.to_datetime("now")
                            ).replace(":", "-").replace(".", "_")
            path = f"./__outputs__/results_{timestamp}.csv"
            df.sort_values(by=["set", "bcubed_fscore"], ascending=[True, False]
                           ).to_csv(path_or_buf=path,
                                    index=False)

    @staticmethod
    def form_problemset_result_dictionary(dictionaries: List[Dict],
                                          l2_norms: List[bool],
                                          identifiers: List[str],
                                          problem_set: int):
        res = defaultdict(list)
        res["set"].extend([problem_set] * len(dictionaries))
        for i, d in enumerate(dictionaries):
            res["algorithm"].append(identifiers[i])
            res["l2_normalised_data"].append(l2_norms[i])
            for k in d.keys():
                res[k].append(d[k])
        return res

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


class AmazonParser:
    """
    Encapsulates parsing and reading Amazon reviews.
    Original source code is available on: http://jmcauley.ucsd.edu/data/amazon/

    """

    @staticmethod
    def _parse(path: str):
        g = gzip.open(path, 'rb')
        for l in g:
            yield eval(l)

    @staticmethod
    def get_dataframe(path: str, columns=["reviewerID", "reviewText"]):
        """
        Reads an Amazon json.gz data file into a pandas dataframe.

        Parameters
        ----------
        path : str
            The path to where the data file is located

        columns : list of str
            The list of columns' names to include in the dataframe

        Examples
        --------
        >>> df = get_dataframe('reviews_Video_Games.json.gz')

        """

        i = 0
        df = {}
        for d in AmazonParser._parse(path):
            df[i] = d
            i += 1

        return pd.DataFrame.from_dict(df, orient='index', columns=columns)


def main():
    print("Aiders here")
    pass


if __name__ == "__main__":
    main()
