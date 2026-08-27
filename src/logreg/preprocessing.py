from os import XATTR_SIZE_MAX
from pathlib import Path
import csv

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

    class_index = 0
    for classe in classes:
        class_grades = [row[class_index] for row in X]
        means.append(list_mean(class_grades))
        stds.append(list_std(class_grades, means[class_index]))
        print(means[class_index], stds[class_index])
        for row in X:
            row[class_index] = (row[class_index] - means[class_index]) / stds[class_index]
        class_index += 1

    return X, means, stds


def apply_preprocessing(X, means, stds):
    ...
