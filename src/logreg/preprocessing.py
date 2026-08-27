from pathlib import Path
import csv

from numpy import isnan

from lib import *


classes = (
    "Astronomy",
    "Herbology",
    "Defense Against the Dark Arts",
    "Divination",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    "Charms",
    "Flying"
)

houses = (
    "Gryffindor",
    "Slytherin",
    "Ravenclaw",
    "Hufflepuff"
)

def load_training_dataset(
    filename: str | Path,
) -> tuple[list[list[float]], list[str]]:
    """
    Load selected features and labels from the given training CSV.

    Returns:
        X: 2D matrix of grades:
        [
            [stud1_grade1, stud1_grade2, ...],
            [stud2_grade1, stud2_grade2, ...],
            ...
        ]

        Y: 1D matrix of houses: [stud1_house, stud2_house, ...]
    """

    X: list[list[float]] = []
    Y: list[str] = []

    with open(filename, "r") as file:
        data = csv.DictReader(file)
        for row in data:
            if row["Hogwarts House"] not in houses:  # if house not defined in row
                continue
            Y.append(row["Hogwarts House"])
            X.append(
                [
                    float(row[classe]) if row[classe] not in (None, "") else nan
                    for classe in classes
                ]
            )
    return (X, Y)


def standardize_grades(
    X: list[list[float]]
) -> tuple[list[list[float]], list[float], list[float]]:
    """
    Standardize all grades from X matrix
    & Gives all stds and means by classes

    Returns:
        X: 2D matrix of standardize grades
        means: list of all classes means
        stds: list of all classes standard deviations
    """
    means: list[float] = []
    stds: list[float] = []

    X_scaled = [row.copy() for row in X]
    class_index = 0
    for classe in classes:
        class_grades = [row[class_index] for row in X_scaled]
        means.append(list_mean(class_grades))
        stds.append(list_std(class_grades, means[class_index]))
        for row in X_scaled:
            if isnan(row[class_index]):
                row[class_index] = means[class_index]
            elif stds[class_index] == 0:
                row[class_index] = 0
            else:
                row[class_index] = (row[class_index] - means[class_index]) / stds[class_index]
        class_index += 1

    return X_scaled, means, stds


def fill_y_vectors(
    Y: list[str]
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Fill the 4 Y vectors (1D matrixes) for each house 
    1 if in house.
    0 if not.

    Returns:
        Y_gryffindor, Y_hufflepuff, Y_ravenclaw, Y_slytherin
    """
    Y_slytherin: list[int] = [0 for elem in Y]
    Y_ravenclaw: list[int] = [0 for elem in Y]
    Y_gryffindor: list[int] = [0 for elem in Y]
    Y_hufflepuff: list[int] = [0 for elem in Y]

    for idx, house in enumerate(Y):
        if house not in houses:
            continue
        if house == "Slytherin":
            Y_slytherin[idx] = 1
        elif house == "Gryffindor":
            Y_gryffindor[idx] = 1
        elif house == "Ravenclaw":
            Y_ravenclaw[idx] = 1
        else:
            Y_hufflepuff[idx] = 1
    return Y_slytherin, Y_ravenclaw, Y_gryffindor, Y_hufflepuff

def add_1_column(
    X_scaled: list[list[float]]
) -> list[list[float]]:
    X_biased: list[list[float]] = []
    for idx, row in enumerate(X_scaled):
        X_biased.append([])
        X_biased[idx].append(1.0)
        X_biased[idx] += row
    return X_biased


def init_weights(
) -> tuple[list[float], list[float], list[float], list[float]]:
    W_slytherin: list[float] = [0.0 for lenght in range(len(classes) + 1)]
    W_ravenclaw: list[float] = [0.0 for lenght in range(len(classes) + 1)]
    W_gryffindor: list[float] =  [0.0 for lenght in range(len(classes) + 1)]
    W_hufflepuff: list[float] = [0.0 for lenght in range(len(classes) + 1)]

    return W_slytherin, W_ravenclaw, W_gryffindor, W_hufflepuff
