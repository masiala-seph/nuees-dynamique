import numpy as np
import matplotlib.pyplot as plt

class DynamicClouds:
    """
    Implémentation de l'algorithme des Nuées Dynamiques (Diday, 1971)
    Variante avec repères/étalons (noyaux de k éléments par classe)
    """
    def __init__(self, n_clusters=3, kernel_size=1, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.kernel_size = kernel_size  # Nombre d'éléments formant le noyau/repère
        self.max_iter = max_iter
        self.tol = tol
        self.kernels = None
        self.labels = None

    def _distance(self, x, kernel):
        # Distance entre un point x et un noyau (moyenne des distances aux points du noyau)
        distances = [np.linalg.norm(x - k_point) for k_point in kernel]
        return np.mean(distances)

    def fit(self, X):
        n_samples, n_features = X.shape
        
        # 1. Initialisation : sélection aléatoire des noyaux (L)
        np.random.seed(42)
        indices = np.random.choice(n_samples, self.n_clusters * self.kernel_size, replace=False)
        self.kernels = [
            X[indices[i * self.kernel_size : (i + 1) * self.kernel_size]]
            for i in range(self.n_clusters)
        ]

        for iteration in range(self.max_iter):
            # 2. Phase d'Affectation (Fonction f) : construction de la partition
            labels = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                distances = [self._distance(X[i], k) for k in self.kernels]
                labels[i] = np.argmin(distances)

            # 3. Phase Représentation (Fonction g) : mise à jour des noyaux
            new_kernels = []
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) == 0:
                    # Gestion des classes vides
                    new_kernels.append(self.kernels[k])
                    continue
                
                # Le nouveau noyau est composé des points du cluster les plus proches du centre de gravité
                center = np.mean(cluster_points, axis=0)
                dists_to_center = [np.linalg.norm(p - center) for p in cluster_points]
                best_indices = np.argsort(dists_to_center)[:min(self.kernel_size, len(cluster_points))]
                new_kernels.append(cluster_points[best_indices])

            # Vérification de la convergence
            kernel_shift = sum(
                np.linalg.norm(np.mean(nk, axis=0) - np.mean(ok, axis=0))
                for nk, ok in zip(new_kernels, self.kernels)
            )
            
            self.kernels = new_kernels
            self.labels = labels

            if kernel_shift < self.tol:
                print(f"Convergence atteinte à l'itération {iteration + 1}")
                break

        return self

    def predict(self, X):
        return np.array([np.argmin([self._distance(x, k) for k in self.kernels]) for x in X])


if __name__ == "__main__":
    # Génération d'un jeu de données de test (3 clusters)
    np.random.seed(0)
    c1 = np.random.randn(100, 2) + np.array([0, 0])
    c2 = np.random.randn(100, 2) + np.array([5, 5])
    c3 = np.random.randn(100, 2) + np.array([0, 5])
    X = np.vstack([c1, c2, c3])

    # Entraînement des nuées dynamiques
    model = DynamicClouds(n_clusters=3, kernel_size=2)
    model.fit(X)

    print("Entraînement terminé avec succès.")