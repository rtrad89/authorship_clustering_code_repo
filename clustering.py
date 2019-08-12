# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 19:25:40 2019

@author: RTRAD
"""
from sklearn.cluster import (MeanShift,
                             AgglomerativeClustering,
                             OPTICS, cluster_optics_dbscan)
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
from numpy import place, column_stack, unique, inf, arange
import pandas as pd
import bcubed
from spherecluster import SphericalKMeans
from gap_statistic import OptimalK
from gmeans import GMeans


class Clusterer:
    # Define algorithms options for external usage
    alg_mean_shift = 1
    alg_h_dbscan = 2
    alg_x_means = 3
    alg_spherical_k_means = 4
    alg_iterative_spherical_k_means = 5
    alg_hac = 6
    alg_optics = 7

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
                self.estimated_k = False
            elif (desired_n_clusters is not None) and (
                    desired_n_clusters < len(dtm)) and (
                            desired_n_clusters > 0):
                self.k = desired_n_clusters
                self.estimated_k = False
            else:
                # Estimate k using on L2 normalised data
                self.k, self.cand_k = self._estimate_k()
                self.estimated_k = True
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
        k_bic = len(unique(self._cluster_xmeans()))

        # Define a custom clusterer for the Gap statistic
        def ms(X, k):
            c = MeanShift()
            c.fit(X)
            return c.cluster_centers_, c.predict(X)
        gap = OptimalK(clusterer=ms)
        k_gap = gap(X=self.data, cluster_array=range(2, len(self.data)-1))

        gmeans = GMeans(random_state=137, max_depth=500)
        gmeans.fit(self.data)
        k_gaussian = len(unique(gmeans.labels_))

        return (round((k_bic + k_gap + k_gaussian) / 3),
                [k_bic, k_gap, k_gaussian])

    def _process_noise_as_singletons(self, result: List):
        place(result, result == -1,
              range(1 + result.max(), 1 + result.max()+len(result))
              )

        return result

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

    def _cluster_spherical_kmeans(self,
                                  init: str = "k-means++"):
        """
        Employ spherical k-means on L2 normalised directional data points. The
        algorithm uses cosine distances internally, and is especially suited
        to textual high dimentional data.

        """

        # Pay attention that k-means++ initialiser may be using Eucledian
        # distances still.. hence the "random" choice
        skm = SphericalKMeans(n_clusters=self.k, init=init, n_init=500,
                              random_state=13712, normalize=False)
        return skm.fit_predict(self.data)

    def _cluster_ispherical_kmeans(self,
                                   init: str = "k-means++"):
        """
        Employ spherical k-means on L2 normalised directional data points in an
        iterative manner to select the best k according to intrinsic clustering
        evaluation measures.

        Parameters
        ----------
        init: str
            The initialisation method - "random" or "k-means++"

        """
        max_sil = -inf
        best_pred = None
        # Pay attention that k-means++ initialiser may be using Eucledian
        # distances still.. hence the "random" choice
        for ik in range(2, len(self.data) - 1):
            skm = SphericalKMeans(n_clusters=ik, init=init, n_init=500,
                                  random_state=13712, normalize=False)
            pred = skm.fit_predict(self.data)
            score = silhouette_score(X=self.data,
                                     metric=self.distance_metric,
                                     labels=pred,
                                     random_state=13712)
            if score > max_sil:
                max_sil = score
                best_pred = pred

        return best_pred

    def _select_best_hac(self,
                         linkage: str,
                         use_sil: bool = True,
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

        for k in range(2, len(self.data)-1):
            hac = AgglomerativeClustering(n_clusters=k,
                                          affinity=self.distance_metric,
                                          linkage=linkage,
                                          compute_full_tree=True,
                                          memory=r"./__cache__/")
            pred = hac.fit_predict(self.data)

            score = silhouette_score(X=self.data,
                                     labels=pred,
                                     metric=self.distance_metric,
                                     random_state=13712)

            if verbose:
                print(f"-► k={k:02} → score={score:.3f}")

            if score > max_score:
                max_score = score
                best_k = k
                clustering = pred
        if verbose:
            print(f"\tBest k for HAC ({linkage}) is {best_k}..")

        return (best_k, max_score, clustering)

    def _cluster_hac(self, linkage: str):
        if self.estimated_k:
            hac_k, _, pred = self._select_best_hac(linkage=linkage,
                                                   verbose=False)
            k = round((hac_k + sum(self.cand_k)) / (1+len(self.cand_k)))
        else:
            k = self.k

        self.cand_k.append(k)
        hac = AgglomerativeClustering(n_clusters=k,
                                      affinity=self.distance_metric,
                                      linkage=linkage)
        pred = hac.fit_predict(self.data)

        return pred

    def _extract_best_optics(self, clusterer):
        max_score = -inf
        best_pred = None

        for my_eps in arange(0.01, 0.5, 0.01):
            pred = cluster_optics_dbscan(
                    reachability=clusterer.reachability_,
                    core_distances=clusterer.core_distances_,
                    ordering=clusterer.ordering_, eps=my_eps)

            if not len(unique(pred)) in (1, len(self.data)):
                score = silhouette_score(X=self.data,
                                         labels=pred,
                                         metric=self.distance_metric,
                                         random_state=13712)

                if score > max_score:
                    max_score = score
                    best_pred = pred

        return self._process_noise_as_singletons(best_pred)

    def _cluster_optics(self):
        optics = OPTICS(min_cluster_size=self.min_clu_size,
                        min_samples=self.min_clu_size,
                        metric=self.distance_metric,
                        leaf_size=len(self.data))
        optics.fit(X=self.data)
        return self._extract_best_optics(optics)

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
                    "Silhouette": round(sil, 4
                                        ) if sil is not None else None,
                    "Calinski_harabasz": round(ch, 4
                                               ) if ch is not None else None,
                    "Davies_Bouldin": round(dv, 4
                                            ) if dv is not None else None
                    # Here goes the unsupervised indices
                    })

        return ret

    def evaluate(self,
                 alg_option: int,
                 param_linkage: str = None,
                 param_init: str = None):

        if alg_option == Clusterer.alg_h_dbscan:
            clustering_lables = self._cluster_hdbscan()

        elif alg_option == Clusterer.alg_hac:
            clustering_lables = self._cluster_hac(
                    linkage=param_linkage)

        elif alg_option == Clusterer.alg_iterative_spherical_k_means:
            clustering_lables = self._cluster_ispherical_kmeans(
                    init=param_init)

        elif alg_option == Clusterer.alg_mean_shift:
            clustering_lables = self._cluster_mean_shift()

        elif alg_option == Clusterer.alg_optics:
            clustering_lables = self._cluster_optics()

        elif alg_option == Clusterer.alg_spherical_k_means:
            clustering_lables = self._cluster_spherical_kmeans()

        elif alg_option == Clusterer.alg_x_means:
            clustering_lables = self._cluster_xmeans()

        else:
            return None

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

    def eval_cluster_ispherical_kmeans(self, init: str = "k-means++"):
        clustering_lables = self._cluster_ispherical_kmeans(init=init)
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

    def eval_cluster_hac(self, linkage: str):
        """
        Execute and evaluate HAC clustering.

        Parameters
        ----------
        k : int
            The number of clusters to extract. If it is not specified then the
            X-Means BIC scheme is exploited for this.
        """

        clustering_lables = self._cluster_hac(linkage=linkage)
        predicted = pd.Series(index=self.data.index, data=clustering_lables,
                              name="predicted")
        aligned_labels = pd.concat([self.true_labels, predicted], axis=1,
                                   sort=False)

        return clustering_lables, self._eval_clustering(
                aligned_labels.true,
                aligned_labels.predicted)

    def eval_cluster_optics(self, sil_scorer: bool = True):
        """
        Execute and evaluate OPTICS clustering.

        """

        clustering_lables = self._cluster_optics(
                silhouette_scorer=sil_scorer)
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
