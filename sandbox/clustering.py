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
                                     normalized_mutual_info_score,
                                     silhouette_score,
                                     calinski_harabasz_score,
                                     davies_bouldin_score)
from typing import List, Dict, Set
from numpy import place, column_stack, unique, inf
import pandas as pd
import bcubed
from spherecluster import SphericalKMeans
from sklearn.preprocessing import normalize
from gap_statistic import OptimalK
from gmeans import GMeans


class Clusterer:

    def __init__(self,
                 dtm: List[List],
                 true_labels: pd.Series,
                 min_cluster_size: int,
                 max_nbr_clusters: int,
                 min_nbr_clusters: int,
                 metric: str,
                 desired_n_clusters: int = None):
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

            if desired_n_clusters == 0:
                # Indicate usage of the best k for testing purposes
                self.k = 1+max(true_labels)
            elif (desired_n_clusters is not None) and (
                    desired_n_clusters < len(dtm)) and (
                            desired_n_clusters > 0):
                self.k = desired_n_clusters
            else:
                # Estimate k using on L2 normalised data
                self.k = self._estimate_k()
        else:
            print("\nERROR: cannot create class.\n"
                  "Data must be passed as a dataframe or similar structure.\n")

    def _estimate_k(self):
        """
        Estimate the best k -number of clusters- using various methods.

        Returns
        -------
        k : int
            An average estimation of three methods: bic, the gap statistic and
            GMeans gauusians.
            Note: the data would be L2-normalised before proceeding

        """

        # Normalise the data to L2 temporarily
        copy_data = self.data.copy()
        self.set_data(pd.DataFrame(normalize(self.data, norm="l2")))
        k_bic = len(unique(self._cluster_xmeans()))

        def ms(X, k):
            c = MeanShift()
            c.fit(X)
            return c.cluster_centers_, c.predict(X)
        gap = OptimalK(clusterer=ms)
        k_gap = gap(X=self.data, cluster_array=range(1, len(self.data)))

        gmeans = GMeans(random_state=137, max_depth=500)
        gmeans.fit(self.data)
        k_gaussian = len(unique(gmeans.labels_))

        # Load the original data again
        self.data = copy_data.copy()
        return round((k_bic + k_gap + k_gaussian) / 3)

    def set_data(self, new_data: List[List]):
        if isinstance(new_data, pd.DataFrame):
            self.data = new_data
        else:
            # Form a dataframe with the same index to be the replacement
            idx = self.data.index
            self.data = pd.DataFrame(data=new_data,
                                     index=idx)

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

    def _cluster_spherical_kmeans(self):
        """
        Employ spherical k-means on L2 normalised directional data points. The
        algorithm uses cosine distances internally, and is especially suited
        to textual high dimentional data.

        """
        # Pay attention that k-means++ initialiser may be using Eucledian
        # distances still.. hence the "random" choice
        skm = SphericalKMeans(n_clusters=self.k, init="k-means++", n_init=25,
                              n_jobs=-1, random_state=13712, normalize=True)
        return skm.fit_predict(self.data)

    def _select_best_hac(self,
                         use_sil: bool = True,
                         linkage: str = "average",
                         verbose: bool = False):
        """
        Select the best cut in the HAC tree according to an unsupervised
        clustering evaluation metric.

        Parameters
        ----------
        k_values : list, optional
            The range of k values to be examined.
        use_sil: bool
            if True, silhouette_score will be used,
            otherwise calinski_harabasz_score is used by default.
        linkage: str
            The type of linkage scheme to be used in the HAC
        """
        max_score = -inf
        best_k = -1
        clustering = None

        for k in range(2, len(self.data)):
            hac = AgglomerativeClustering(n_clusters=k,
                                          affinity=self.distance_metric,
                                          linkage=linkage,
                                          compute_full_tree=True,
                                          memory=r"./__cache__/")
            pred = hac.fit_predict(self.data)
            if use_sil:
                score = silhouette_score(X=self.data,
                                         labels=pred,
                                         metric=self.distance_metric,
                                         random_state=13712)
            else:
                score = calinski_harabasz_score(
                        X=self.data, labels=pred)

            if verbose:
                print(f"-► k={k:02} → score={score:.3f}")

            if score > max_score:
                max_score = score
                best_k = k
                clustering = pred

        return (best_k, max_score, clustering)

    def _cluster_hac(self, k: int, linkage: str = "average"):
        if k is None:
            # Compute the full tree and cache it to extract the best clustering
#            results = self._select_best_hac(
#                    k_values=range(2, 1+len(self.data)//2),
#                    use_sil=False)
            # The results do not seem to be useful... Terminating this path
            hac = AgglomerativeClustering(n_clusters=self.k,
                                          affinity=self.distance_metric,
                                          linkage=linkage)
        else:
            hac = AgglomerativeClustering(n_clusters=k,
                                          affinity=self.distance_metric,
                                          linkage=linkage)

        return hac.fit_predict(self.data)

    def _hdp_topic_clusters(self):
        """
        Convert the topic ascriptions to deterministic clusters using max

        Returns
        -------
        pd.Series
            The indexed deterministic clustering of the hdp soft clustering.
            The instance is assigned to the cluster which has the maximum
            weight.
        """
        return self.data.idxmax(axis=1)


# =============================================================================
#     def _cluster_agglomerative(self, k: int, linkage: str = "complete"):
#         """
#         Use SciPy implementation of HAC to avail myself of the automatice cut
#         location detection.
#         """
#         # Define the linkage scheme
#         z = hac.linkage(self.data.to_numpy(),
#                         metric="cosine",
#                         method=linkage)
#         return hac.fcluster(Z=z, t=k, criterion="maxclust")
# =============================================================================

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

        # =====================================================================
        # Unsupervised Metrics
        # =====================================================================
        if not labels_predicted.nunique() in (1, len(self.data)):
            sil = silhouette_score(X=self.data,
                                   labels=labels_predicted,
                                   metric=self.distance_metric,
                                   random_state=13712)

            ch = calinski_harabasz_score(X=self.data, labels=labels_predicted)

            dv = davies_bouldin_score(X=self.data, labels=labels_predicted)
        else:
            sil = None
            ch = None
            dv = None

        ret = {}
        ret.update({"nmi": round(nmi, 4),
                    "ami": round(ami, 4),
                    "ari": round(ari, 4),
                    "fms": round(fms, 4),
                    "v_measure": round(v_measure, 4),
                    "bcubed_precision": round(bcubed_precision, 4),
                    "bcubed_recall": round(bcubed_recall, 4),
                    "bcubed_fscore": round(bcubed_f1, 4),
                    "AVERAGE": round(
                            (nmi+ami+ari+fms+v_measure+bcubed_f1)/6,
                            2),
                    "Silhouette": round(sil, 4
                                        ) if sil is not None else None,
                    "Calinski_harabasz": round(ch, 4
                                               ) if ch is not None else None,
                    "Davies_Bouldin": round(dv, 4
                                            ) if dv is not None else None
                    # Here goes the unsupervised indices
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

    def eval_cluster_spherical_kmeans(self):
        clustering_lables = self._cluster_spherical_kmeans()
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

# =============================================================================
#     def eval_cluster_agglomerative(self, linkage="complete"):
#         """
#         Execute and evaluate HAC clustering.
#
#         Parameters
#         ----------
#         k : int
#             The number of clusters to extract. If it is not specified then
#             the X-Means BIC scheme is exploited for this.
#         """
#
#         clustering_lables = self._cluster_agglomerative(k=self.k,
#                                                         linkage=linkage)
#         predicted = pd.Series(index=self.data.index, data=clustering_lables,
#                               name="predicted")
#         aligned_labels = pd.concat([self.true_labels, predicted], axis=1,
#                                    sort=False)
#
#         return clustering_lables, self._eval_clustering(
#                 aligned_labels.true,
#                 aligned_labels.predicted)
#
# =============================================================================
    def eval_cluster_hac(self, k=None, linkage="complete"):
        """
        Execute and evaluate HAC clustering.

        Parameters
        ----------
        k : int
            The number of clusters to extract. If it is not specified then the
            X-Means BIC scheme is exploited for this.
        """

        clustering_lables = self._cluster_hac(k=k, linkage=linkage)
        predicted = pd.Series(index=self.data.index, data=clustering_lables,
                              name="predicted")
        aligned_labels = pd.concat([self.true_labels, predicted], axis=1,
                                   sort=False)

        return clustering_lables, self._eval_clustering(
                aligned_labels.true,
                aligned_labels.predicted)

    def eval_cluster_hdp(self):
        predicted = self._hdp_topic_clusters()
        predicted.name = "predicted"

        aligned_labels = pd.concat([self.true_labels, predicted], axis=1,
                                   sort=False)

        return predicted.to_numpy(), self._eval_clustering(
                aligned_labels.true,
                aligned_labels.predicted)

    def eval_true_clustering(self):
        return self.true_labels.to_numpy(), self._eval_clustering(
                self.true_labels,
                self.true_labels)


def main():
    exit(0)


if __name__ == "__main__":
    main()
