from math import sqrt
import math
from numpy import isnan, nan


def list_sum(L: list[float]) -> float:
    res = 0
    for i in L:
        res += float(i)
    return res


def list_mean(L: list[float]) -> float:
    if L is None or len(L) == 0:
        return 0.0
    mean = 0.0
    nan_cpt = 0
    for elem in L:
        if not isnan(elem):
            mean += elem
        else:
            nan_cpt += 1
    mean /= len(L) - nan_cpt
    return mean


def list_std(L: list[float], mean: float | None = None) -> float:
    if mean is None or mean == 0:
        mean = list_mean(L)
    res = 0
    nan_cpt = 0
    for val in L:
        if not isnan(val):
            res += (val - mean) ** 2
        else:
            nan_cpt += 1
    res /= (len(L) - 1 - nan_cpt)
    return sqrt(res)


def first_quartile(sorted_list: list[float]) -> float:
    return sorted_list[round(len(sorted_list) / 4)]


def median(sorted_list: list[float]) -> float:
    if len(sorted_list) % 2 == 1:
        return sorted_list[len(sorted_list) // 2]
    else:
        return (
            sorted_list[len(sorted_list) // 2]
            + sorted_list[(len(sorted_list) // 2) - 1]
        ) / 2


def third_quartile(sorted_list: list[float]) -> float:
    return sorted_list[round(len(sorted_list) * 3 / 4)]


def min_idx(L: list[float]) -> int:
    min = L[0]
    id = 0
    for idx, elem in enumerate(L):
        if elem < min:
            min = elem
            id = idx
    return id


def sigmoid(Z: float) -> float:
    """
    - sigmoid(z) = 1 / (1 + exp(-z))
    - This will turn the linear combination z into a probability
    """

    if Z >= 0:
        return 1 / (1 + math.exp(-Z))

    exp_z = math.exp(Z)
    return exp_z / (1 + exp_z)


def dot_product(
    matrix: list[list[float]],
    vector: list[float]
) -> list[float]:
    if len(matrix) == 0 or len(vector) != len(matrix[0]):
        return []
    Z: list[float] = [0.0 for elem in matrix]
    for row_index, row in enumerate(matrix):
        for idx, coef in enumerate(vector):
            Z[row_index] += row[idx] * coef
    return Z


def transpose_matrix(
    matrix: list[list[float]]
) -> list[list[float]]:
    """
    Returns:
        transposed version of given matrix
        [3, 4, 5]       [3, 1]
        [1, 2 ,6]   =>  [4, 2]
                        [5, 6]
    """
    if not matrix:
        return []
    T: list[list[float]] = [[] for elem in matrix[0]]
    for row in matrix:
        for idx, elem in enumerate(row):
            T[idx].append(elem)
    return T
