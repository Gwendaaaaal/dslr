from math import sqrt

def list_sum(L: list[float]) -> float:
    res = 0
    for i in L:
        res += float(i)
    return res

def list_mean(L: list[float]) -> float:
    mean = 0.0
    for elem in L:
        mean += elem
    mean /= len(L)
    return (mean)

def list_std(L: list[float], mean: float | None = None) -> float:
    if mean is None :
        mean = list_mean(L)
    res = 0
    for i in L :
        res += (i - mean) ** 2
    res = res / (len(L) - 1)
    return sqrt(res)

def first_quartile(sorted_list: list[float]) -> float:
    return sorted_list[round(len(sorted_list) / 4)]

def median(sorted_list: list[float]) -> float:
    if (len(sorted_list) % 2 == 1):
        return sorted_list[len(sorted_list) // 2]
    else:
        return ((sorted_list[len(sorted_list) // 2] + sorted_list[(len(sorted_list) // 2) - 1]) / 2)

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
