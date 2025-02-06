from abc import ABC, abstractmethod
import numpy as np

# Clase Point
class Point:
    def __init__(self, ndim, values):
        self.ndim = ndim
        self.values = values

    def get_ndim(self):
        return self.ndim

    def get_values(self):
        return self.values

    def __str__(self):
        return f"Point(dim={self.ndim}, values={self.values})"

# Clase abstracta Distance
class Distance(ABC):
    @abstractmethod
    def compute_distance(self, point1, point2):
        pass

# Distancia Manhattan
class ManhattanDistance(Distance):
    def compute_distance(self, point1, point2):
        return np.sum(np.abs(np.array(point1.get_values()) - np.array(point2.get_values())))

# Distancia Euclidiana
class EuclideanDistance(Distance):
    def compute_distance(self, point1, point2):
        return np.sqrt(np.sum((np.array(point1.get_values()) - np.array(point2.get_values()))**2))

# Distancia Chebyshev
class ChebyshevDistance(Distance):
    def compute_distance(self, point1, point2):
        return np.max(np.abs(np.array(point1.get_values()) - np.array(point2.get_values())))

# Distancia Canberra
class CanberraDistance(Distance):
    def compute_distance(self, point1, point2):
        values1 = np.array(point1.get_values())
        values2 = np.array(point2.get_values())
        return np.sum(np.abs(values1 - values2) / (np.abs(values1) + np.abs(values2) + 1e-10))

# Distancia Mahalanobis
class MahalanobisDistance(Distance):
    def compute_distance(self, point1, point2):
        values1 = np.array(point1.get_values())
        values2 = np.array(point2.get_values())
        covariance_matrix = np.cov(np.stack((values1, values2)).T)
        inv_cov_matrix = np.linalg.inv(covariance_matrix + 1e-10 * np.eye(len(covariance_matrix)))
        diff = values1 - values2
        return np.sqrt(diff.T @ inv_cov_matrix @ diff)

# Distancia Wasserstein
class WassersteinDistance(Distance):
    def compute_distance(self, point1, point2):
        values1 = np.sort(point1.get_values())
        values2 = np.sort(point2.get_values())
        return np.sum(np.abs(values1 - values2)) / len(values1)

if __name__ == "__main__":
    # Puntos de prueba
    p1 = Point(3, [1, 2, 3])
    p2 = Point(3, [4, 5, 6])

    # Instanciar clases de distancia
    distances = [
        ManhattanDistance(),
        EuclideanDistance(),
        ChebyshevDistance(),
        CanberraDistance(),
        MahalanobisDistance(),
        WassersteinDistance()
    ]

    # Calcular y mostrar distancias
    for distance in distances:
        print(f"{distance.__class__.__name__}: {distance.compute_distance(p1, p2)}")
