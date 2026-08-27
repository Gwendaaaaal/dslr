import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from numpy import nan

from preprocessing import load_training_dataset, standardize_grades


def logreg_train(filename: str | Path) -> None:
    """
    logreg_train takes dataset_train.csv as a parameter.
    For the mandatory part, you must use the technique of gradient descent to minimize the error.
    The program generates a file containing the weights that will be used for the prediction
    """
    # ============================================================
    # STEP 1: LOAD DATA
    # ============================================================
    # - Read dataset_train.csv (e.g. with pandas)
    # - Separate into:
    #     X = the feature columns you selected (Herbology, Divination,
    #         Muggle Studies, Ancient Runes, History of Magic,
    #         Transfiguration, Potions, Charms, Flying, Astronomy)
    #     y = the "Hogwarts House" column (the labels)

    unprocessed_X: list[list[float]]
    Y: list[str]
    unprocessed_X, Y = load_training_dataset(filename)

    # ============================================================
    # STEP 3: NORMALIZE / STANDARDIZE FEATURES
    # ============================================================
    # - For each feature column, compute:
    #     mean = average of the column (on training data only)
    #     std  = standard deviation of the column (on training data only)
    # - Store these mean/std values somewhere ( needed in logreg_predict
    #   AND needed to save them in weights file)
    # - Transform X: X_scaled = (X - mean) / std
    X_scaled, means, stds = standardize_grades(unprocessed_X)
    print(X_scaled)

    # ============================================================
    # STEP 4: PREPARE LABELS FOR ONE-VS-ALL
    # ============================================================
    # - Get the list of unique classes (the 4 houses)
    # - For each house, create a binary label vector:
    #     y_house = 1 if student's house == this house, else 0
    # - You'll end up with 4 separate binary label vectors
    #   (Gryffindor vs rest, Slytherin vs rest, Ravenclaw vs rest, Hufflepuff vs rest)

    # ============================================================
    # STEP 5: ADD BIAS TERM
    # ============================================================
    # - Add a column of 1s to X_scaled (this multiplies with w0, the bias)
    # - This lets you treat w0 like any other weight in your dot product

    # ============================================================
    # STEP 6: INITIALIZE WEIGHTS
    # ============================================================
    # - For each of the 4 house models, initialize a weight vector
    #   (length = number of features + 1 for bias), e.g. all zeros

    # ============================================================
    # STEP 7: DEFINE SIGMOID FUNCTION
    # ============================================================
    # - sigmoid(z) = 1 / (1 + exp(-z))
    # - This will turn your linear combination z into a probability

    # ============================================================
    # STEP 8: GRADIENT DESCENT LOOP (repeat for EACH of the 4 houses)
    # ============================================================
    # For each house model:
    #   Repeat for a fixed number of iterations (or until convergence):
    #     a. Compute z = X_scaled_with_bias . weights   (dot product)
    #     b. Compute predictions: h = sigmoid(z)
    #     c. Compute the error: error = h - y_house
    #     d. Compute the gradient:
    #          gradient = (1/m) * X_scaled_with_bias.T . error
    #        (m = number of training examples)
    #     e. Update weights:
    #          weights = weights - learning_rate * gradient
    #     f. (Optional but recommended) Compute and store the cost
    #        (log loss) at each iteration so you can plot it later
    #        and check that it's decreasing (sanity check for debugging)

    # ============================================================
    # STEP 9: SAVE RESULTS
    # ============================================================
    # - Save to a file (e.g. weights.csv or weights.json):
    #     - The 4 weight vectors (one per house)
    #     - The mean and std used for normalization (needed by logreg_predict)
    #     - The list of feature column names IN THE ORDER you used them
    #       (so logreg_predict knows exactly which columns to extract, and in what order)
    #     - The list of house names IN THE ORDER of your weight vectors
    #       (so logreg_predict knows which weight vector belongs to which house)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display statistics for the numerical features of a dataset."
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to the CSV dataset",
    )
    args = parser.parse_args()

    logreg_train(args.dataset)


if __name__ == "__main__":
    main()
