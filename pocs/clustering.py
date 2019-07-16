# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 19:25:40 2019

@author: RTRAD
"""
from sklearn.cluster import DBSCAN, OPTICS
import hdbscan
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.metrics.cluster import (adjusted_mutual_info_score,
                                     adjusted_rand_score,
                                     v_measure_score,
                                     fowlkes_mallows_score,
                                     normalized_mutual_info_score)
from typing import List, Dict, Set
from numpy import place
import pandas as pd
import bcubed


class Clusterer:

    def __init__(self,
                 dtm: List[List],
                 true_labels: pd.Series,
                 min_cluster_size: int,
                 max_nbr_clusters: int,
                 min_nbr_clusters: int,
                 metric: str):
        """
        The default constructor, encapsulating common attributes

        Parameters
        ----------
        metric : str
            The metric to use when calculating distance between instances
            in a feature array.It must be one of the options allowed by
            sklearn.metrics.pairwise_distances for its metric parameter.
        """
        if isinstance(dtm, pd.DataFrame):
            self.data = dtm
            self.true_labels = true_labels
            self.min_clu_size = min_cluster_size
            self.min_clusters = min_nbr_clusters
            self.max_clusters = max_nbr_clusters
            self.distance_metric = metric
        else:
            print("\nERROR: cannot create class.\n"
                  "Data must be passed as a dataframe or similar structure.\n")

    def set_data(self, data):
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            print("\nERROR while setting new data!\n"
                  "Data must be passed as a dataframe or similar structure.\n")

    def _process_noise_as_singletons(self, result: List):

        place(result, result == -1,
              range(1 + result.max(), 1 + result.max()+len(result))
              )

        return result

    def _cluster_dbscan(self, epsilon: float, min_pts: int):
        labels = DBSCAN(eps=epsilon, min_samples=min_pts,
                        metric=self.distance_metric, n_jobs=-1
                        ).fit_predict(self.data)

        return self._process_noise_as_singletons(labels)

    def _cluster_optics(self):
        """
        https://scikit-learn.org/stable/auto_examples/cluster/plot_optics.html#sphx-glr-auto-examples-cluster-plot-optics-py
        """
        pass

    def _cluster_hdbscan(self):
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
        # Since HDBSCAN discards noisy docs, we will convert them into
        # singleton clusters

        return self._process_noise_as_singletons(result=result)

    def _cluster_xmeans(self) -> list:
        """
        Perform partitional centroid-based x-means

        .. _Documentation:
            https://pyclustering.github.io/docs/0.9.0/html/index.html

        """

        # Use Kmeans++ technique to initialise cluster centers
        initial_centers = kmeans_plusplus_initializer(
                self.data.to_numpy(), 2).initialize()
        # Set the maximum number of clusters to half the count of data points
        xmeans_instance = xmeans(self.data.to_numpy(),
                                 initial_centers, self.max_clusters)
        xmeans_instance.process()

        # Extract the clusters and return them
        return xmeans_instance.get_clusters()

    def _reshape_labels_as_dicts(
            self, labels: pd.Series) -> Dict[str, Set[str]]:
        if not(isinstance(labels, pd.Series)):
            print("Input must be a pandas series with an index.")
            return None
        # Convert the series to a dict of set of labels
        return pd.Series(index=labels.index,
                         data=[set(str(v)) for v in labels.values]).to_dict()

    def eval_clustering(self, labels_true, labels_predicted):
        nmi = normalized_mutual_info_score(labels_true,
                                           labels_predicted,
                                           average_method="max")

        ami = adjusted_mutual_info_score(labels_true,
                                         labels_predicted,
                                         average_method="arithmetic")

        ari = adjusted_rand_score(labels_true,
                                  labels_predicted)

        v_measure = v_measure_score(labels_true,
                                    labels_predicted,
                                    beta=1.0)

        fms = fowlkes_mallows_score(labels_true,
                                    labels_predicted)

        # Reshape labels for BCubed measures
        true_dict = self._reshape_labels_as_dicts(labels_true)
        pred_dict = self._reshape_labels_as_dicts(labels_predicted)

        bcubed_precision = bcubed.precision(cdict=pred_dict, ldict=true_dict)
        bcubed_recall = bcubed.recall(cdict=pred_dict, ldict=true_dict)
        bcubed_f1 = bcubed.fscore(bcubed_precision, bcubed_recall)

        ret = {}
        ret.update({"nmi": nmi,
                    "ami": ami,
                    "ari": ari,
                    "fms": fms,
                    "v_measure": v_measure,
                    "bcubed_precision": bcubed_precision,
                    "bcubed_recall": bcubed_recall,
                    "bcubed_fscore": bcubed_f1,
                    "AVG": (nmi+ami+ari+fms+v_measure+bcubed_f1)/6})

        return ret

    def eval_cluster_dbscan(self, epsilon: float, min_pts: int):
        clustering_lables = self._cluster_dbscan(epsilon=epsilon,
                                                 min_pts=min_pts)
        predicted = pd.Series(index=self.data.index, data=clustering_lables,
                              name="predicted")
        aligned_labels = pd.concat([self.true_labels, predicted], axis=1,
                                   sort=False)

        return clustering_lables, self.eval_clustering(
                aligned_labels.true,
                aligned_labels.predicted)


def main():
    pass


if __name__ == "__main__":
    main()
