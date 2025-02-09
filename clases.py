from abc import ABC, abstractmethod
import numpy as np


class Point:
    '''
    Clase Point que representa un punto en un espacio de dimensión n

    Parameters
    ----------
    ndim : int
        Dimensión del espacio
    values : list
        Valores de las coordenadas

    methods
    -------
    get_ndim()
        Devuelve la dimensión del espacio
    get_values()
        Devuelve los valores de las coordenadas
    '''

    def __init__(self, ndim, values):
        '''Constructor de la clase Point'''
        self.ndim = ndim
        self.values = values

    def get_ndim(self):
        '''Devuelve la dimensión del espacio'''
        return self.ndim

    def get_values(self):
        '''Devuelve los valores de las coordenadas'''
        return self.values

    def __str__(self):
        '''Devuelve una representación de la clase'''
        return f'Point(dim={self.ndim}, values={self.values})'


class Distance(ABC):
    '''
    Clase abstracta Distance que representa una distancia
    methods
    -------
    compute_distance(point1, point2)
        Devuelve la distancia entre dos puntos
    '''
    @abstractmethod
    def compute_distance(self, point1, point2):
        pass


class ManhattanDistance(Distance):
    '''
    Clase ManhattanDistance que representa la distancia de Manhattan
    methods
    -------
    compute_distance(point1, point2)
    '''

    def compute_distance(self, point1, point2):
        '''Devuelve la distancia de Manhattan'''
        return np.sum(np.abs(np.array(point1.get_values()) -
                      np.array(point2.get_values())))


class EuclideanDistance(Distance):
    '''
    Clase EuclideanDistance que representa la distancia euclideana
    methods
    -------
    compute_distance(point1, point2)
    '''

    def compute_distance(self, point1, point2):
        '''Devuelve la distancia euclideana'''
        return np.sqrt(
            np.sum(
                (np.array(
                    point1.get_values()) -
                    np.array(
                    point2.get_values()))**2))


class ChebyshevDistance(Distance):
    '''
    Clase ChebyshevDistance que representa la distancia de Chebyshev
    methods
    -------
    compute_distance(point1, point2)
    '''

    def compute_distance(self, point1, point2):
        '''Devuelve la distancia de Chebyshev'''
        return np.max(np.abs(np.array(point1.get_values()) -
                      np.array(point2.get_values())))


class CanberraDistance(Distance):
    '''
    clase CanberraDistance que representa la distancia de Canberra
    methods
    -------
    compute_distance(point1, point2)
    '''

    def compute_distance(self, point1, point2):
        '''Devuelve la distancia de Canberra'''
        values1 = np.array(point1.get_values())
        values2 = np.array(point2.get_values())
        return np.sum(np.abs(values1 - values2) /
                      (np.abs(values1) + np.abs(values2) + 1e-10))


class MahalanobisDistance(Distance):
    '''
    clase MahalanobisDistance que representa la distancia de Mahalanobis
    methods
    -------
    compute_distance(point1, point2)
    '''

    def compute_distance(self, point1, point2):
        '''Devuelve la distancia de Mahalanobis'''
        values1 = np.array(point1.get_values())
        values2 = np.array(point2.get_values())
        covariance_matrix = np.cov(np.stack((values1, values2)).T)
        inv_cov_matrix = np.linalg.inv(
            covariance_matrix +
            1e-10 *
            np.eye(
                len(covariance_matrix)))
        diff = values1 - values2
        return np.sqrt(diff.T @ inv_cov_matrix @ diff)


class WassersteinDistance(Distance):
    '''Clase WassersteinDistance que representa la distancia de Wasserstein'''

    def compute_distance(self, point1, point2):
        '''Devuelve la distancia de Wasserstein'''
        values1 = np.sort(point1.get_values())
        values2 = np.sort(point2.get_values())
        return np.sum(np.abs(values1 - values2)) / len(values1)
