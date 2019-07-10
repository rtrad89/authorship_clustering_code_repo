# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 19:25:40 2019

@author: RTRAD
"""
import pandas as pd
import hdbscan
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer


class AuthorClusterer:

    def __init__(self, lables: list, dtm: pd.DataFrame):
        self.gound_truth = lables
        self.data = dtm

    def cluster_hdbscan():
        """
        Perform density based hierarchical DBSCAN

        References
        ----------
            https://pyclustering.github.io/docs/0.9.0/html/index.html#intro_sec
        """
        pass

    def cluster_xmeans(data: list) -> list:
        """Perform partitional centroid-based x-means"""
        # Use Kmeans++ technique to initialise cluster centers
        initial_centers = kmeans_plusplus_initializer(
                data, 2).initialize()
        # Set the maximum number of clusters to half the count of data points
        xmeans_instance = xmeans(data, initial_centers, len(data)//2)
        xmeans_instance.process()

        # Extract the clusters and return them
        return xmeans_instance.get_clusters()

    def adjusted_rand_evaluate():
        pass

    def v_measure_evalualte():
        pass
