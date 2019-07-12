# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 19:25:40 2019

@author: RTRAD
"""
import hdbscan
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.metrics.cluster import (adjusted_mutual_info_score,
                                     adjusted_rand_score)
from typing import List


class Clusterer:

    def __init__(self,
                 dtm: List[List],
                 label_vec: List,
                 min_cluster_size: int,
                 max_nbr_clusters: int,
                 min_nbr_clusters: int):
        """The default constructor"""

        self.data = dtm
        self.labels_vec = label_vec
        self.min_clu_size = min_cluster_size
        self.min_clusters = min_nbr_clusters
        self.max_clusters = max_nbr_clusters

    def cluster_hdbscan(self):
        """
        Perform density based hierarchical DBSCAN

        Returns
        -------
        result : numpy.ndarray
            The clustering in an array, where first entry corresponds to first
            document in self.data

        .. _Documentation:
            https://github.com/scikit-learn-contrib/hdbscan

        """

        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_clu_size)
        result = clusterer.fit_predict(self.data)

        return result

    def cluster_xmeans(self) -> list:
        """
        Perform partitional centroid-based x-means

        .. _Documentation:
            https://pyclustering.github.io/docs/0.9.0/html/index.html

        """

        # Use Kmeans++ technique to initialise cluster centers
        initial_centers = kmeans_plusplus_initializer(
                self.data, 2).initialize()
        # Set the maximum number of clusters to half the count of data points
        xmeans_instance = xmeans(self.data,
                                 initial_centers, self.max_clusters)
        xmeans_instance.process()

        # Extract the clusters and return them
        return xmeans_instance.get_clusters()

    def adjusted_mutual_info_eval(self, clustering: list):
        """
        Computes the AMI score

        .. _Documentation:
            https://scikit-learn.org/stable/modules/clustering.html

        """

        return adjusted_mutual_info_score(self.labels_vec,
                                          clustering)

    def adjusted_rand_index_eval(self, clustering: list):
        """
        Computes the ARI score

        .. _Documentation:
            https://scikit-learn.org/stable/modules/clustering.html

        """

        return adjusted_rand_score(self.labels_vec, clustering)


def main():
    pass


if __name__ == "__main__":
    main()
