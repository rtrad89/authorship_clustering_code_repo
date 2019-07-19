# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 19:25:40 2019

@author: RTRAD
"""
from sklearn.cluster import DBSCAN, MeanShift, AgglomerativeClustering
import hdbscan
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from sklearn.metrics.cluster import (adjusted_mutual_info_score,
                                     adjusted_rand_score,
                                     v_measure_score,
                                     fowlkes_mallows_score,
                                     normalized_mutual_info_score)
from typing import List, Dict, Set
from numpy import place, column_stack
import pandas as pd
import bcubed
from spherecluster import SphericalKMeans

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

    def _cluster_mean_shift(self):
        return MeanShift().fit_predict(self.data)

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
        # Initialise HDBSCAN, using generic algorithms to allow for cosine
        # distances (due to technical incompatability with sklearn). Also the
        # min_samples is set to 1 because we want to declare as few points to
        # be noise as possible
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_clu_size,
                                    metric=self.distance_metric,
                                    min_samples=1,
                                    algorithm="generic")
        result = clusterer.fit_predict(self.data)
        # Since HDBSCAN discards noisy docs, we will convert them into
        # singleton clusters

        return self._process_noise_as_singletons(result=result)

    def _process_xmeans_output(self, clustering: List[List]) -> list:
        """Convert the list of lists that xmeans output to a indexed list"""
        vals = [d for sublist in clustering for d in sublist]
        keys = [item for sublist in
                [[i]*len(clustering[i]) for i, _ in enumerate(clustering)]
                for item in sublist]
        # Stack them as columns
        c = column_stack((keys, vals))
        # Sort data by documents' indices, so that the labels correspond to th-
        # em in order
        c = c[c[:, 1].argsort(kind="mergesort")]
        # Return the sorted labels
        return c[:, 0]

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
        xmeans_instance = xmeans(data=self.data.to_numpy(),
                                 initial_centers=initial_centers,
                                 kmax=self.max_clusters)
        xmeans_instance.process()

        # Extract the clusters and return them
        return self._process_xmeans_output(xmeans_instance.get_clusters())

    def _cluster_spherical_kmeans(self, k=None):
        """
        Employ spherical k-means on L2 normalised directional data points. The
        algorithm uses cosine distances internally, and is especially suited
        to textual high dimentional data.

        """
        # Select the best k depending on xmeans BIC model selection
        # CAUTION: xmeans uses Eucledian distances! may be incompatible
        if k is None:
            k = len(self._cluster_xmeans())
        # Pay attention that k-means++ initialiser may be using Eucledian
        # distances still.. hence the "random" choice
        skm = SphericalKMeans(n_clusters=k, init="k-means++", n_init=25,
                              n_jobs=-1, random_state=13712, normalize=True)
        return skm.fit_predict(self.data)

    def _reshape_labels_as_dicts(
            self, labels: pd.Series) -> Dict[str, Set[str]]:
        if not(isinstance(labels, pd.Series)):
            print("Input must be a pandas series with an index.")
            return None
        # Convert the series to a dict of set of labels
        return pd.Series(index=labels.index,
                         data=[set(str(v)) for v in labels.values]).to_dict()

    def _eval_clustering(self, labels_true, labels_predicted):
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
        ret.update({"nmi": round(nmi, 4),
                    "ami": round(ami, 4),
                    "ari": round(ari, 4),
                    "fms": round(fms, 4),
                    "v_measure": round(v_measure, 4),
                    "bcubed_precision": round(bcubed_precision, 4),
                    "bcubed_recall": round(bcubed_recall, 4),
                    "bcubed_fscore": round(bcubed_f1, 4)
                    })

        return ret

    def eval_cluster_dbscan(self, epsilon: float, min_pts: int):
        clustering_lables = self._cluster_dbscan(epsilon=epsilon,
                                                 min_pts=min_pts)
        predicted = pd.Series(index=self.data.index, data=clustering_lables,
                              name="predicted")
        aligned_labels = pd.concat([self.true_labels, predicted], axis=1,
                                   sort=False)

        return clustering_lables, self._eval_clustering(
                aligned_labels.true,
                aligned_labels.predicted)

    def eval_cluster_hdbscan(self):
        clustering_lables = self._cluster_hdbscan()
        predicted = pd.Series(index=self.data.index, data=clustering_lables,
                              name="predicted")
        aligned_labels = pd.concat([self.true_labels, predicted], axis=1,
                                   sort=False)

        return clustering_lables, self._eval_clustering(
                aligned_labels.true,
                aligned_labels.predicted)

    def eval_cluster_spherical_kmeans(self, k=None):
        clustering_lables = self._cluster_spherical_kmeans(k)
        predicted = pd.Series(index=self.data.index, data=clustering_lables,
                              name="predicted")
        aligned_labels = pd.concat([self.true_labels, predicted], axis=1,
                                   sort=False)

        return clustering_lables, self._eval_clustering(
                aligned_labels.true,
                aligned_labels.predicted)

    def eval_cluster_mean_shift(self):
        clustering_lables = self._cluster_mean_shift()
        predicted = pd.Series(index=self.data.index, data=clustering_lables,
                              name="predicted")
        aligned_labels = pd.concat([self.true_labels, predicted], axis=1,
                                   sort=False)

        return clustering_lables, self._eval_clustering(
                aligned_labels.true,
                aligned_labels.predicted)

    def eval_cluster_xmeans(self):
        """
        Evaluate the clustering of X-Means. Although it uses Euclidean metrics
        under the hood, if the data is L2 normalised on the sample level, i.e.
        the samples are projected onto an n-sphere, Euclidean distances resemb-
        le angular distances now, a close approximation of the cosine distance
        which suits directional data more.
        """
        clustering_lables = self._cluster_xmeans()
        predicted = pd.Series(index=self.data.index, data=clustering_lables,
                              name="predicted")
        aligned_labels = pd.concat([self.true_labels, predicted], axis=1,
                                   sort=False)

        return clustering_lables, self._eval_clustering(
                aligned_labels.true,
                aligned_labels.predicted)


def main():
    pass


if __name__ == "__main__":
    main()
