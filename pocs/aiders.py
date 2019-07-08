# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:27:46 2019

@author: RTRAD
"""
import os
from shutil import rmtree
import pandas as pd
import gzip


class DiskTools:
    """A helper class providing methods for managing files and folders"""

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
            print("> ERROR: Please make sure the {} folder"
                  "is not used by some process").format(dir_path)


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
    pass


if __name__ == "__main__":
    main()
