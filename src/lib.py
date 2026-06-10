from math import sqrt

def list_sum(L: list[float]) -> float:
    res = 0
    for i in L:
        res += float(i)
    return res

def list_std(L: list[float], mean: float | None) -> float:
    if mean == None :
        mean = 0.0 # TODO: calculate mean
    res = 0;
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
