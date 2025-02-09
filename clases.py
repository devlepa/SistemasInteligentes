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
        '''
        Constructor de la clase Point
        '''
        self.ndim = ndim
        self.values = values

    def get_ndim(self):
        '''
        Devuelve la dimensión del espacio
        '''
        return self.ndim

    def get_values(self):
        '''
        Devuelve los valores de las coordenadas
        '''
        return self.values

    def __str__(self):
        '''
        Devuelve una representación de la clase
        '''
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
        '''
        Devuelve la distancia de Manhattan
        '''
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
        '''
        Devuelve la distancia euclideana
        '''
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
        '''
        Devuelve la distancia de Chebyshev
        '''
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
        '''
        devuelve la distancia de Canberra
        '''
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
        '''
        devuelve la distancia de Mahalanobis
        '''
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
    '''
    clase WassersteinDistance que representa la distancia de Wasserstein
    '''
    def compute_distance(self, point1, point2):
        '''
        devuelve la distancia de Wasserstein
        '''
        values1 = np.sort(point1.get_values())
        values2 = np.sort(point2.get_values())
        return np.sum(np.abs(values1 - values2)) / len(values1)
    

#from clases import Point, ManhattanDistance, EuclideanDistance, ChebyshevDistance, CanberraDistance, MahalanobisDistance, WassersteinDistance
#import numpy as np
import matplotlib.pyplot as plt

# Función para ingresar puntos
def input_point(ndim):
    print(f"Ingrese las coordenadas del punto en {ndim} dimensiones (separadas por espacio):")
    coords = list(map(float, input().split()))
    if len(coords) != ndim:
        raise ValueError(f"Debes ingresar exactamente {ndim} coordenadas.")
    return coords

# Función para graficar puntos
def plot_points(point1, point2):
    if point1.get_ndim() == 2:
        # Gráfico en 2D
        plt.figure(figsize=(6, 6))
        plt.scatter(point1.get_values()[0], point1.get_values()[1], color='red', label='Punto 1')
        plt.scatter(point2.get_values()[0], point2.get_values()[1], color='blue', label='Punto 2')
        plt.plot([point1.get_values()[0], point2.get_values()[0]],
                 [point1.get_values()[1], point2.get_values()[1]], color='green', linestyle='--')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Visualización de puntos en 2D')
        plt.legend()
        plt.grid()
        plt.show()
    elif point1.get_ndim() == 3:
        # Gráfico en 3D
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(point1.get_values()[0], point1.get_values()[1], point1.get_values()[2], color='red', label='Punto 1')
        ax.scatter(point2.get_values()[0], point2.get_values()[1], point2.get_values()[2], color='blue', label='Punto 2')
        ax.plot([point1.get_values()[0], point2.get_values()[0]],
                [point1.get_values()[1], point2.get_values()[1]],
                [point1.get_values()[2], point2.get_values()[2]], color='green', linestyle='--')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Visualización de puntos en 3D')
        plt.legend()
        plt.show()
    else:
        print("La visualización gráfica solo está disponible para 2 o 3 dimensiones.")

# Solicitar al usuario la dimensión del espacio
ndim = int(input("¿En cuántas dimensiones trabajarás? (Introduce un número entero): "))
if ndim <= 0:
    raise ValueError("La dimensión debe ser un número entero positivo.")

# Ingresar las coordenadas de los puntos
print("Punto 1:")
point1 = Point(ndim, input_point(ndim))
print("Punto 2:")
point2 = Point(ndim, input_point(ndim))

# Crear instancias de las clases de distancia
distances = {
    "Manhattan": ManhattanDistance(),
    "Euclidiana": EuclideanDistance(),
    "Chebyshev": ChebyshevDistance(),
    "Canberra": CanberraDistance(),
    "Mahalanobis": MahalanobisDistance(),
    "Wasserstein": WassersteinDistance()
}

# Calcular y mostrar las distancias
print("\nCálculos de distancias:")
for name, distance in distances.items():
    try:
        result = distance.compute_distance(point1, point2)
        print(f"{name}: {result}")
    except Exception as e:
        print(f"{name}: Error al calcular la distancia ({e})")

# Mostrar puntos para verificar
print("\nResumen de puntos ingresados:")
print(f"Punto 1: {point1}")
print(f"Punto 2: {point2}")

# Graficar los puntos si es posible
plot_points(point1, point2)

