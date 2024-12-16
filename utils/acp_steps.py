import numpy as np

def normalize_data(data):
    """
    Centre et réduit les données.
    Chaque colonne est normalisée pour avoir une moyenne nulle et un écart-type de 1.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std

def compute_correlation_matrix(data):
    """
    Calcule la matrice de corrélation.
    Chaque colonne représente une variable.
    """
    return np.corrcoef(data, rowvar=False)

def compute_eigenvalues_eigenvectors(correlation_matrix):
    """
    Calcule les valeurs propres et les vecteurs propres.
    Retourne les vecteurs propres triés par ordre décroissant des valeurs propres.
    """
    eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)

    # Tri des valeurs et vecteurs propres par ordre décroissant des valeurs propres
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    return eigenvalues, eigenvectors

