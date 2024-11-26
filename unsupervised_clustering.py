import os

# Set the number of threads for OpenMP
os.environ["OMP_NUM_THREADS"] = "2"

import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (
    AgglomerativeClustering,
    KMeans,
    DBSCAN, SpectralClustering, HDBSCAN, OPTICS, Birch, AffinityPropagation, BisectingKMeans, MeanShift
)
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import (
    adjusted_mutual_info_score, adjusted_rand_score, calinski_harabasz_score,
    silhouette_score, silhouette_samples, davies_bouldin_score, completeness_score,
    fowlkes_mallows_score, homogeneity_score, v_measure_score, mutual_info_score,
    normalized_mutual_info_score, rand_score, homogeneity_completeness_v_measure,
)
from sklearn.metrics.cluster import contingency_matrix, pair_confusion_matrix
import seaborn as sns
from scipy.spatial.distance import cdist


class ClusteringAnalysis:
    def __init__(self, data, random_state=42, ground_truth=None):
        """
        Initialize the clustering analysis with data

        Parameters:
        data: pandas DataFrame or numpy array
        random_state: int, for reproducibility
        """
        self.raw_data = data
        self.data = StandardScaler().fit_transform(data)
        self.random_state = random_state
        self.ground_truth = ground_truth

    def calculate_cluster_metrics(self, labels):
        """
        Calculate various clustering metrics

        Parameters:
        labels: array, cluster labels

        # +-----------------------------------------+------------------------------------------------------+
        # | Metric/Function                         | Description                                          |
        # +-----------------------------------------+------------------------------------------------------+
        # | adjusted_mutual_info_score              | Adjusted Mutual Information between two clusterings. |
        # | adjusted_rand_score                     | Rand index adjusted for chance.                      |
        # | calinski_harabasz_score                 | Compute the Calinski and Harabasz score.             |
        # | cluster.contingency_matrix              | Build a contingency matrix for label relationships.  |
        # | cluster.pair_confusion_matrix           | Pair confusion matrix from two clusterings.          |
        # | completeness_score                      | Compute completeness metric with ground truth.       |
        # | davies_bouldin_score                    | Compute the Davies-Bouldin score.                    |
        # | fowlkes_mallows_score                   | Similarity measure between two clusterings.          |
        # | homogeneity_completeness_v_measure      | Compute homogeneity, completeness, and V-Measure.    |
        # | homogeneity_score                       | Homogeneity metric with ground truth.                |
        # | mutual_info_score                       | Mutual Information between two clusterings.          |
        # | normalized_mutual_info_score            | Normalized Mutual Information between clusterings.   |
        # | rand_score                              | Rand index.                                          |
        # | silhouette_samples                      | Silhouette Coefficient for each sample.              |
        # | silhouette_score                        | Mean Silhouette Coefficient of all samples.          |
        # | v_measure_score                         | V-measure cluster labeling with ground truth.        |
        # +-----------------------------------------+------------------------------------------------------+

        """
        metrics = {
            # Unsupervised metrics
            'silhouette_score': silhouette_score(self.data, labels),
            'calinski_harabasz_score': calinski_harabasz_score(self.data, labels),
            'davies_bouldin_score': davies_bouldin_score(self.data, labels),

            # Supervised metrics (if ground truth is provided)
            'adjusted_rand_score': adjusted_rand_score(self.ground_truth,
                                                       labels) if self.ground_truth is not None else None,
            'adjusted_mutual_info_score': adjusted_mutual_info_score(self.ground_truth,
                                                                     labels) if self.ground_truth is not None else None,
            'completeness_score': completeness_score(self.ground_truth,
                                                     labels) if self.ground_truth is not None else None,
            'fowlkes_mallows_score': fowlkes_mallows_score(self.ground_truth,
                                                           labels) if self.ground_truth is not None else None,
            'homogeneity_score': homogeneity_score(self.ground_truth,
                                                   labels) if self.ground_truth is not None else None,
            'v_measure_score': v_measure_score(self.ground_truth, labels) if self.ground_truth is not None else None,
            'mutual_info_score': mutual_info_score(self.ground_truth,
                                                   labels) if self.ground_truth is not None else None,
            'normalized_mutual_info_score': normalized_mutual_info_score(self.ground_truth,
                                                                         labels) if self.ground_truth is not None else None,
            'rand_score': rand_score(self.ground_truth, labels) if self.ground_truth is not None else None,

            # Pairwise metrics
            'contingency_matrix': contingency_matrix(self.ground_truth,
                                                     labels).tolist() if self.ground_truth is not None else None,
            'pair_confusion_matrix': pair_confusion_matrix(self.ground_truth,
                                                           labels).tolist() if self.ground_truth is not None else None,

            # Combined metrics
            'homogeneity_completeness_v_measure': homogeneity_completeness_v_measure(self.ground_truth,
                                                                                     labels) if self.ground_truth is not None else None,
        }

        # Remove None values (if ground truth is not provided)
        metrics = {key: value for key, value in metrics.items() if value is not None}
        return metrics

    def compute_ssd(self, labels, n_clusters):
        ssd = 0
        for k in range(n_clusters):
            cluster_points = self.data[labels == k]  # Points in cluster k
            centroid = cluster_points.mean(axis=0)  # Centroid of cluster k
            ssd += np.sum((cluster_points - centroid) ** 2)  # SSD for cluster k
        return ssd / np.var(self.data)

    def plot_clusters(self, labels, title, centroids=None):
        # Plot results
        plt.figure(figsize=(10, 7))

        categories = np.unique(labels)

        unique_categories = list(set(categories))
        colors = matplotlib.pyplot.get_cmap(name='viridis',
                                            lut=len(unique_categories))  # Use 'tab10' or any other colormap
        color_mapping = {category: colors(i) for i, category in enumerate(unique_categories)}

        # Map categories to colors
        plt.scatter(self.raw_data.iloc[:, 0],
                    self.raw_data.iloc[:, 1],
                    c=[color_mapping[cat] for cat in labels])
        plt.title(f'{title} clustering')
        plt.legend(handles=[
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_mapping[cat], label=cat)
            for cat in unique_categories
        ], title='Categories')

        if centroids is not None:
            plt.scatter(centroids[:, 0], centroids[:, 1],
                        marker='x', s=200, linewidths=3,
                        color='r', label='Centroids')
        plt.show()

    def hierarchical_clustering(self, n_clusters=3, linkage_method='ward'):
        """
        Perform hierarchical clustering with different linkage methods

        Parameters:
        n_clusters: int, number of clusters
        linkage_method: str, linkage method ('ward', 'complete', 'average', 'single', 'centroid')
        """
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method
        )
        labels = clustering.fit_predict(self.data)

        self.plot_clusters(labels, "Hierarchical ({})".format(linkage_method))

        # Calculate and print metrics
        metrics = self.calculate_cluster_metrics(labels)
        print(f"\nHierarchical Clustering ({linkage_method}) Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.3f}")
            else:
                print(f"{metric}: {value}")

        return labels

    def kmeans_clustering(self, n_clusters=3, algorithm='lloyd'):
        """
        Perform k-means clustering with different implementations

        Parameters:
        n_clusters: int, number of clusters
        algorithm: str, 'lloyd' or 'elkan'
        """
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=self.random_state,
                        algorithm=algorithm)
        labels = kmeans.fit_predict(self.data)
        centroids = kmeans.cluster_centers_

        self.plot_clusters(labels, centroids=centroids, title=f'K-means (k={n_clusters})')

        # Calculate and print metrics
        metrics = self.calculate_cluster_metrics(labels)
        print(f"\nK-means Clustering ({algorithm}) Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")

        return labels, centroids

    def gaussian_mixture_clustering(self, n_components=3, covariance_type='full'):
        """
        Perform Gaussian Mixture Model clustering with EM algorithm

        Parameters:
        n_components: int, number of Gaussian components
        covariance_type: str, type of covariance parameters ('full', 'tied', 'diag', 'spherical')
        """
        gmm = GaussianMixture(n_components=n_components,
                              covariance_type=covariance_type,
                              random_state=self.random_state)
        labels = gmm.fit_predict(self.data)

        centroids = gmm.means_

        # Plot results
        self.plot_clusters(labels, centroids=centroids, title=f'Gaussian Mixture Model Clustering ({covariance_type})')

        # Plot probability contours
        x, y = np.mgrid[self.raw_data.iloc[:, 0].min():self.raw_data.iloc[:, 0].max():100j,
               self.raw_data.iloc[:, 1].min():self.raw_data.iloc[:, 1].max():100j]
        positions = np.vstack([x.ravel(), y.ravel()]).T

        # Transform positions to match the scaling of training data
        scaler = StandardScaler()
        scaler.fit(self.raw_data)
        positions_scaled = scaler.transform(positions)

        probabilities = gmm.predict_proba(positions_scaled)

        plt.figure(figsize=(10, 7))
        for i in range(n_components):
            plt.contour(x, y, probabilities[:, i].reshape(100, 100), levels=[0.5])
        scatter = plt.scatter(self.raw_data.iloc[:, 0],
                              self.raw_data.iloc[:, 1],
                              c=labels,
                              cmap='viridis')
        plt.title(f'GMM Probability Contours ({covariance_type})')
        plt.colorbar(scatter)
        plt.show()

        # Calculate and print metrics
        metrics = self.calculate_cluster_metrics(labels)
        print(f"\nGMM Clustering ({covariance_type}) Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")

        return labels, gmm.means_, gmm.covariances_

    def spectral_clustering(
            self,
            n_clusters=3,
            affinity='nearest_neighbors',
            random_state=42
    ):
        """
        Perform Spectral Clustering

        Args:
            n_clusters (int): Number of clusters
            affinity (str): Method to construct affinity matrix
            random_state (int): Seed for reproducibility

        Returns:
            numpy.ndarray: Cluster labels
        """
        model = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            random_state=random_state
        )
        return model.fit_predict(self.data)

    def mean_shift(
            self,
            bandwidth=None,
            bin_seeding=True,
            cluster_all=True
    ):
        """
        Perform Mean Shift Clustering

        Args:
            bandwidth (float): Kernel bandwidth. If None, estimated from data
            bin_seeding (bool): Use binning to reduce computational complexity
            cluster_all (bool): If True, all points are clustered

        Returns:
            numpy.ndarray: Cluster labels
        """
        model = MeanShift(
            bandwidth=bandwidth,
            bin_seeding=bin_seeding,
            cluster_all=cluster_all
        )
        return model.fit_predict(self.data)

    def dbscan(
            self,
            eps=0.5,
            min_samples=5,
            metric='euclidean'
    ):
        """
        Perform DBSCAN Clustering

        Args:
            eps (float): Maximum distance between two samples to be considered in same neighborhood
            min_samples (int): Minimum number of samples in a neighborhood
            metric (str): Distance metric

        Returns:
            numpy.ndarray: Cluster labels
        """
        model = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric
        )
        return model.fit_predict(self.data)

    def hdbscan(
            self,
            min_cluster_size=5,
            min_samples=None,
            metric='euclidean'
    ):
        """
        Perform HDBSCAN Clustering

        Args:
            min_cluster_size (int): Minimum number of points to form a cluster
            min_samples (int): Minimum number of points in a core sample
            metric (str): Distance metric

        Returns:
            numpy.ndarray: Cluster labels
        """
        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric
        )
        return model.fit_predict(self.data)

    def optics(
            self,
            min_samples=5,
            max_eps=np.inf,
            metric='minkowski'
    ):
        """
        Perform OPTICS Clustering

        Args:
            min_samples (int): Minimum number of samples in a neighborhood
            max_eps (float): Maximum distance between two samples
            metric (str): Distance metric

        Returns:
            numpy.ndarray: Cluster labels
        """
        model = OPTICS(
            min_samples=min_samples,
            max_eps=max_eps,
            metric=metric
        )
        return model.fit_predict(self.data)

    def birch(
            self,
            threshold=0.5,
            branching_factor=50,
            n_clusters=3
    ):
        """
        Perform BIRCH Clustering

        Args:
            threshold (float): Radius of the subcluster
            branching_factor (int): Maximum number of CF subclusters in each node
            n_clusters (int): Number of clusters after the tree is built

        Returns:
            numpy.ndarray: Cluster labels
        """
        model = Birch(
            threshold=threshold,
            branching_factor=branching_factor,
            n_clusters=n_clusters
        )
        return model.fit_predict(self.data)

    def bisecting_kmeans(
            self,
            n_clusters=3,
            random_state=42
    ):
        """
        Perform Bisecting K-Means Clustering

        Args:
            n_clusters (int): Number of clusters
            random_state (int): Seed for reproducibility

        Returns:
            numpy.ndarray: Cluster labels
        """
        model = BisectingKMeans(
            n_clusters=n_clusters,
            random_state=random_state
        )
        return model.fit_predict(self.data)

    def affinity_propagation(
            self,
            damping=0.5,
            preference=None,
            random_state=42
    ):
        """
        Perform Affinity Propagation Clustering

        Args:
            damping (float): Damping factor between 0.5 and 1
            preference (array-like, optional): Preferences for each point
            random_state (int): Seed for reproducibility

        Returns:
            numpy.ndarray: Cluster labels
        """
        model = AffinityPropagation(
            damping=damping,
            preference=preference,
            random_state=random_state
        )
        return model.fit_predict(self.data)

    def find_optimal_k(self, max_k=10, method='silhouette', clustering_f=None):
        """
        Find optimal number of clusters using various methods

        Parameters:
        max_k: int, maximum number of clusters to test
        method: str, method to use ('silhouette', 'elbow', 'gap')
        """
        if clustering_f is None:
            clustering_f = self.kmeans_clustering
        scores = []
        k_values = range(2, max_k + 1)
        plt.figure(figsize=(10, 7))

        if method == 'silhouette':
            for k in k_values:
                out = clustering_f(k)
                if isinstance(out, tuple):
                    labels = out[0]
                else:
                    labels = out
                score = silhouette_score(self.data, labels)
                scores.append(score)
            plt.title('Silhouette Score vs Number of Clusters')
            plt.ylabel('Silhouette Score')

        elif method == 'elbow':
            for k in k_values:
                out = clustering_f(k)
                if isinstance(out, tuple):
                    labels = out[0]
                else:
                    labels = out
                score = self.compute_ssd(labels=labels, n_clusters=k)
                scores.append(score)
            plt.title('Elbow Method')
            plt.ylabel('SSD')

        # Plot results

        plt.plot(k_values, scores, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.grid(True)
        plt.show()

        if method == 'silhouette':
            optimal_k = k_values[np.argmax(scores)]
        else:  # elbow method
            # Find the elbow point using the maximum curvature
            diffs = np.diff(scores, 2)
            optimal_k = k_values[np.argmax(np.abs(diffs)) + 1]

        return optimal_k

    def compare_methods(self, n_clusters=None):
        """
        Compare different clustering methods

        Parameters:
        n_clusters: int, number of clusters
        """

        # Perform all clustering methods with different variations
        results = {}

        # Hierarchical clustering with different linkage methods
        for linkage in ['ward', 'complete', 'average', 'single']:
            if n_clusters is None:
                k = self.find_optimal_k(clustering_f=lambda x: self.hierarchical_clustering(x, linkage))
                labels = self.hierarchical_clustering(k, linkage)
            else:
                labels = self.hierarchical_clustering(n_clusters, linkage)
            results[f'Hierarchical ({linkage})'] = self.calculate_cluster_metrics(labels)

        # K-means with different implementations
        for algo in ['lloyd', 'elkan']:
            if n_clusters is None:
                k = self.find_optimal_k(clustering_f=lambda x: self.kmeans_clustering(x, algo))
                labels, _ = self.kmeans_clustering(k, algo)
            else:
                labels, _ = self.kmeans_clustering(n_clusters, algo)
            results[f'K-means ({algo})'] = self.calculate_cluster_metrics(labels)

        # GMM with different covariance types
        for cov_type in ['full', 'tied', 'diag', 'spherical']:
            if n_clusters is None:
                k = self.find_optimal_k(clustering_f=lambda x: self.gaussian_mixture_clustering(x, cov_type))
                labels, _, _ = self.gaussian_mixture_clustering(k, cov_type)
            else:
                labels, _, _ = self.gaussian_mixture_clustering(n_clusters, cov_type)
            results[f'GMM ({cov_type})'] = self.calculate_cluster_metrics(labels)

        # Plot comparison
        metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))

        for i, metric in enumerate(metrics):
            scores = [result[metric] for result in results.values()]
            axes[i].bar(results.keys(), scores)
            axes[i].set_title(f'{metric.replace("_", " ").title()} Score')
            axes[i].tick_params(axis='x', rotation=45)

            # Set custom ylim for better readability
            min_value = min(scores)  # Replace `values` with your data
            max_value = max(scores)  # Replace `values` with your data
            # Define margin for ylim
            margin = 0.1 * (max_value - min_value)  # 10% of the range

            axes[i].set_ylim(
                min_value - margin,
                max_value + margin
            )

        plt.tight_layout()
        plt.show()

        return results

def get_normal_clusters(n_samples=300):
    # Create three clusters with different shapes
    cluster1 = np.random.normal(loc=[0, 0], scale=0.5, size=(n_samples // 3, 2))
    cluster2 = np.random.normal(loc=[2, 2], scale=0.5, size=(n_samples // 3, 2))
    cluster3 = np.random.normal(loc=[0, 2], scale=0.5, size=(n_samples // 3, 2))

    # Combine clusters
    X = np.vstack([cluster1, cluster2, cluster3])
    y = np.concatenate([np.ones(cluster1.shape[0]), 2 * np.ones(cluster2.shape[0]), 3 * np.ones(cluster3.shape[0])])
    return X, y

def get_normalish_clusters(n_samples=300):
    # Create stretched/rotated normal-like clusters
    cluster1 = np.dot(np.random.normal(size=(n_samples // 3, 2)), [[1.5, 0.2], [0.2, 0.5]]) + [0, 0]
    cluster2 = np.dot(np.random.normal(size=(n_samples // 3, 2)), [[0.5, -0.4], [-0.4, 1.5]]) + [3, 1]
    cluster3 = np.dot(np.random.normal(size=(n_samples // 3, 2)), [[1.0, 0.3], [0.3, 0.7]]) + [-2, 4]

    # Combine clusters
    X = np.vstack([cluster1, cluster2, cluster3])
    y = np.concatenate([np.ones(cluster1.shape[0]), 2 * np.ones(cluster2.shape[0]), 3 * np.ones(cluster3.shape[0])])
    return X, y


def get_spiral_clusters(n_samples=300, noise=0.2):
    n = n_samples // 3  # Samples per spiral
    t = np.linspace(0, 3 * np.pi, n)  # Angle range for spirals
    c = 0.3

    # Base spirals
    spiral1 = np.array([c * t * np.cos(t), t * np.sin(t)]).T + np.random.normal(scale=noise, size=(n, 2))
    spiral2 = np.array([t * np.cos(t), c * t * np.sin(t)]).T + np.random.normal(scale=noise, size=(n, 2))
    spiral3 = np.array([c * t * np.cos(t), c * t * np.sin(t)]).T + np.random.normal(scale=noise, size=(n, 2))

    # Apply transformations
    rotation_matrix = lambda theta: np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    # Spiral 2: Rotate by 45° and translate
    spiral2 = spiral2 @ rotation_matrix(np.pi / 4) + [5, 4]

    # Spiral 3: Rotate by -90° and translate
    spiral3 = spiral3 @ rotation_matrix(-np.pi / 2) + [-6, 2]

    # Combine spirals
    X = np.vstack([spiral1, spiral2, spiral3])
    y = np.concatenate([np.ones(n), 2 * np.ones(n), 3 * np.ones(n)])

    ca = ClusteringAnalysis(pd.DataFrame(X, columns=['Feature1', 'Feature2']))
    ca.plot_clusters(y, "Ground truth")
    return X, y


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 300

    X,y = get_spiral_clusters(n_samples=n_samples)

    df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])

    # Initialize clustering analysis
    ca = ClusteringAnalysis(df, ground_truth=y)

    ca.compare_methods(n_clusters=None)
