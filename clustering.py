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
import random
from collections import defaultdict
from cop_kmeans import cop_kmeans
from sys import exit


class Clusterer:
    # Define algorithms options for external usage
    alg_mean_shift = 1
    alg_h_dbscan = 2
    alg_x_means = 3
    alg_spherical_k_means = 4
    alg_iterative_spherical_k_means = 5
    alg_hac = 6
    alg_optics = 7
    bl_random = 8
    bl_singleton = 9
    alg_cop_kmeans = 10

    def __init__(self,
                 dtm: List[List],
                 true_labels: pd.Series,
                 min_cluster_size: int,
                 max_nbr_clusters: int,
                 min_nbr_clusters: int,
                 metric: str,
                 desired_n_clusters: int = None,
                 include_gap: bool = True,
                 include_bic: bool = False):
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
            self.gap = include_gap
            self.bic = include_bic

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
                # Estimate k using on L2 normalised data in spherical k-means
                self.estimated_k = True
        else:
            print("\nERROR: cannot create class.\n"
                  "Data must be passed as a dataframe or the likes.\n")

    def _estimate_k(self, include_bic: bool, include_gap: bool):
        """
        Estimate the best k -number of clusters- using various methods.

        Returns
        -------
        k : int
            An average estimation of three methods: bic, the gap statistic and
            GMeans gauusians.
            Note: the data would be L2-normalised before proceeding

        """
        gmeans = GMeans(random_state=None, max_depth=500)
        gmeans.fit(self.data)
        k_gaussian = len(unique(gmeans.labels_))

        if include_gap:
            # Define a custom clusterer for the Gap statistic
            def ms(X, k):
                c = MeanShift()
                c.fit(X)
                return c.cluster_centers_, c.predict(X)
            gap = OptimalK(clusterer=ms)
            k_gap = gap(X=self.data, cluster_array=range(2, len(self.data)-1))

            if include_bic:
                k_bic = len(unique(self._cluster_xmeans()))
                est_k = round((k_bic + k_gap + k_gaussian) / 3)
                return (est_k,
                        [est_k, k_bic, k_gap, k_gaussian])
            else:
                est_k = round((k_gap + k_gaussian) / 2)
                return (est_k,
                        [est_k, k_gap, k_gaussian])
        # TODO: None gap stats would generate errors when averaging
        # to form k-trends
        else:
            if include_bic:
                k_bic = len(unique(self._cluster_xmeans()))
                est_k = round((k_bic + k_gaussian) / 3)
                return (est_k,
                        [est_k, k_bic, None, k_gaussian])
            else:
                est_k = round(k_gaussian)
                return (est_k,
                        [est_k, None, k_gaussian])

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
                                  init: str = "k-means++",
                                  runs: int = 5):
        """
        Employ spherical k-means on L2 normalised directional data points. The
        algorithm uses cosine distances internally, and is especially suited
        to textual high dimentional data. Since it exploits randomness in
        initialisations and estimations of k, we avergare over 10 runs to
        increase the reliability and reproducibility of results.
        Attention: This routine also estimates k and stores it in the Clusterer

        Returns
        -------
            A array of clusterings with many runs to estimate k
        """
        # For proper saving of k_vals, this must be called first
        # Pay attention that k-means++ initialiser may be using Eucledian
        # distances still.. but l2 norms approx
        preds = []
        if self.estimated_k:
            cands = []
            for rc in range(0, runs):
                # Get the estimated final k,
                # and the Gap and GMeans values in cand
                n, cand = self._estimate_k(include_bic=self.bic,
                                           include_gap=self.gap)
                # Append the three values
                cands.append(cand)
                skm = SphericalKMeans(n_clusters=n, init=init,
                                      random_state=None, normalize=False)
                preds.append(skm.fit_predict(self.data))

            # Append the average of candidate k over runs
            self.cand_k = list(pd.np.mean(cands, axis=0))
            self.k = round(self.cand_k[0])
            # Because round(np.float) is float not int, we need to coerce int:
            if type(self.k) is not int:
                self.k = int(self.k)
        else:
            for rc in range(0, runs):
                skm = SphericalKMeans(n_clusters=self.k, init=init,
                                      random_state=None, normalize=False)
                preds.append(skm.fit_predict(self.data))

        return preds

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
        # distances still..
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
            self.cand_k.append(hac_k)
        else:
            hac_k = self.k

        hac = AgglomerativeClustering(n_clusters=hac_k,
                                      affinity=self.distance_metric,
                                      linkage=linkage)
        pred = hac.fit_predict(self.data)

        return pred

    def _extract_best_optics(self, clusterer):
        max_score = -inf
        best_pred = None

        # Traverse epsilon to detect the best cut
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

        if best_pred is not None:
            return self._process_noise_as_singletons(best_pred)
        else:
            # All outputs are either one cluster or n clusters
            return self._process_noise_as_singletons(pred)

    def _cluster_optics(self):
        optics = OPTICS(min_cluster_size=self.min_clu_size,
                        min_samples=self.min_clu_size,
                        metric=self.distance_metric,
                        leaf_size=len(self.data))
        optics.fit(X=self.data)
        pred = self._extract_best_optics(optics)
        # Append its k to the list of values
        if self.estimated_k:
            self.cand_k.append(1+max(pred))
        return pred

    def _elicit_random_constraints(self,
                                   truth: pd.Series,
                                   prct: float = 0.05):
        """
        Generate ML and NL constraints from the ground truth for COPKMeans
        """
        n = len(truth)
        links_space_size = .5 * n * (n-1)
        required_links = round(prct * links_space_size)
        pairs = set()
        random.seed(13712)
        while (len(pairs) < required_links):
            linking_docs = tuple(random.sample(range(0, len(truth)), 2))
            # Avoid the other permutation of the same link
            i_linking_docs = (linking_docs[1], linking_docs[0])
            if i_linking_docs not in pairs:
                pairs.add(linking_docs)
        # Build the must-link and cannot-link sets
        must_link, cannot_link = [], []
        for p in pairs:
            d1, d2 = p
            must_be_linked = (truth[[d1, d2]].nunique() == 1)
            if must_be_linked:
                must_link.append(p)
            else:
                cannot_link.append(p)
        return must_link, cannot_link, pairs

    def _elicit_author_based_const(self,
                                   truth: pd.Series):
        pass

    def _cluster_constraint_based_estimate_k(self,
                                             constraints_size: float = 0.05,
                                             initialisation: str = "random",
                                             random_const: bool = True):
        """
        A controller that runs COP KMEANS n-2 times and selects the best model.

        Parameters
        ----------
        random_const : bool
            True whether constraints are to be generated randomly;
            False if constraints are generated author-wise.

        Returns
        -------
        list
            The clustering of the best COP KMEANS chosen via grid search..

        """
        # Generate the constraints once
        truth = self.true_labels.sort_index()
        if random_const:
            must_l, cant_l, _ = self._elicit_random_constraints(
                truth=truth,
                prct=constraints_size)
        else:
            must_l, cant_l, _ = self._elicit_author_based_const(
                truth=truth)

        # If k is to be estimated:
        if self.estimated_k:
            stats = []

            for k in range(2, len(self.data)):
                try:
                    pred, _ = cop_kmeans(dataset=self.data.to_numpy(),
                                         k=k,
                                         ml=must_l,
                                         cl=cant_l,
                                         initialization=initialisation)
                    # Calculate DB index
                    if pred is not None:  # Flagging a failure to cluster
                        dbi = davies_bouldin_score(X=self.data, labels=pred)
                        sil = silhouette_score(X=self.data, labels=pred)
                        stats.append([k, dbi, sil, pred])
                except IndexError:
                    print("\t\tGrid Search Early Termination: "
                          f"Attempting COP-KMEANS with k={k} was unworkable.")
                    break

            # Detect the best k and the best clustering based on DB index
            df_stats = pd.DataFrame(stats, columns=["k", "dbi", "sil", "pred"])
            # Calculate the penalised score per each k
            df_stats["score"] = df_stats.dbi * df_stats.k
            # Pick the lowest db score
            best_record = df_stats[df_stats.score == df_stats.score.min()]

            # If more than one optimum is there,
            # opt for the smallest k as it is less likely to be an overfit
            best_record = best_record[best_record.k == best_record.k.min()]
            self.cand_k.append(int(best_record.k))

            return best_record.pred.tolist()[0]
        else:
            pred, _ = cop_kmeans(dataset=self.data.to_numpy(),
                                 k=self.k,
                                 ml=must_l,
                                 cl=cant_l,
                                 initialization=initialisation)
            return pred

    def _bl_random(self):
        rand_k = random.randint(1, len(self.data) + 1)
        # Assign doucments on random
        return random.choices(
            population=list(range(rand_k)),
            k=len(self.data))

    def _bl_singleton(self):
        return list(range(len(self.data)))

    def _hdp_topic_clusters(self):
        """Convert the topic ascriptions to deterministic clusters using max.

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
                         data=[set([str(v)]) for v in labels.values]).to_dict()

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
                 param_init: str = None,
                 param_constraints_size: float = 0.05,
                 param_copkmeans_init: str = "random",
                 param_copkmeans_random_constraints: bool = True):

        if alg_option == Clusterer.alg_spherical_k_means:
            # Here an array will be returned
            clustering_labels = self._cluster_spherical_kmeans(
                    init=param_init)

        elif alg_option == Clusterer.alg_cop_kmeans:
            clustering_labels = self._cluster_constraint_based_estimate_k(
                    constraints_size=param_constraints_size,
                    initialisation=param_copkmeans_init,
                    random_const=param_copkmeans_random_constraints)

        elif alg_option == Clusterer.alg_h_dbscan:
            clustering_labels = self._cluster_hdbscan()

        elif alg_option == Clusterer.alg_hac:
            clustering_labels = self._cluster_hac(
                    linkage=param_linkage)

        elif alg_option == Clusterer.alg_iterative_spherical_k_means:
            clustering_labels = self._cluster_ispherical_kmeans(
                    init=param_init)

        elif alg_option == Clusterer.alg_mean_shift:
            clustering_labels = self._cluster_mean_shift()

        elif alg_option == Clusterer.alg_optics:
            clustering_labels = self._cluster_optics()

        elif alg_option == Clusterer.alg_x_means:
            clustering_labels = self._cluster_xmeans()

        elif alg_option == Clusterer.bl_random:
            clustering_labels = self._bl_random()

        elif alg_option == Clusterer.bl_singleton:
            clustering_labels = self._bl_singleton()

        else:
            return None

        if clustering_labels is not None:
            if type(clustering_labels[0]) is not pd.np.ndarray:
                predicted = pd.Series(index=self.data.index,
                                      data=clustering_labels,
                                      name="predicted")
                aligned_labels = pd.concat([self.true_labels, predicted],
                                           axis=1,
                                           sort=False)
                return clustering_labels, self._eval_clustering(
                        aligned_labels.true,
                        aligned_labels.predicted)
            else:
                # We have a list of predictions of many runs of an algo
                res = defaultdict(list)
                for c in range(0, len(clustering_labels)):
                    predicted = pd.Series(index=self.data.index,
                                          data=clustering_labels[c],
                                          name="predicted")
                    aligned_labels = pd.concat([self.true_labels, predicted],
                                               axis=1,
                                               sort=False)
                    evals = self._eval_clustering(
                            aligned_labels.true,
                            aligned_labels.predicted)
                    for k, v in evals.items():
                        res[k].append(v)

                # TODO: can the following be converted to a dict comprehension?
                m_res = {}
                for k, v in res.items():
                    if None in v:
                        m_res[k] = None
                    else:
                        m_res[k] = pd.np.mean(v)

                # Return one sample clustering and the average expected scores
                pred = clustering_labels[
                        random.randint(0, len(clustering_labels)-1)]
                return pred, m_res

        else:
            return None, None

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

    def eval_sota(self, sota_predicted: pd.Series):
        return sota_predicted.to_numpy(), self._eval_clustering(
                labels_predicted=sota_predicted,
                labels_true=self.true_labels)


def main():
    exit(0)


if __name__ == "__main__":
    main()
