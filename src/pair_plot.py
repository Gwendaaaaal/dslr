import csv
import argparse
from pathlib import Path
from lib import list_std, list_mean, min_idx
import matplotlib.pyplot as plt

def pair_plot(filename: str | Path) -> None:
    classes = {
        "Arithmancy": 0,
        "Astronomy": 1,
        "Herbology": 2,
        "Defense Against the Dark Arts": 3,
        "Divination": 4,
        "Muggle Studies": 5,
        "Ancient Runes": 6,
        "History of Magic": 7,
        "Transfiguration": 8,
        "Potions": 9,
        "Care of Magical Creatures": 10,
        "Charms": 11,
        "Flying": 12,
    }

    houses = {"Gryffindor" : "firebrick", "Slytherin" : "green", "Ravenclaw" : "royalblue", "Hufflepuff" : "gold"}

    # 1 array / house / class 
    grades = [ {
        "Gryffindor": [],
        "Slytherin": [],
        "Ravenclaw": [],
        "Hufflepuff": []
    } for _ in classes]

    with open(filename, "r") as file:
        data = csv.DictReader(file)
        for row in data:
            house = row["Hogwarts House"]
            if house not in houses: #if house not defined in row
                continue
            for classe, idx in classes.items():
                value = row[classe]
                grades[idx][house].append(float(value) if value else None)

    fig, axes = plt.subplots(13, 13, figsize=(30, 20))

    std_of_mean_by_houses = [0.0] * len(classes)

    index_graph = 0
    for classe in classes:
        for comp_class in classes:
            ax = axes.flat[index_graph]
            for house in houses:
                if classe == comp_class:
                    data = [g for g in grades[classes[classe]][house] if g is not None]
                    ax.hist(
                            data,
                            bins=20,
                            alpha=0.5,
                            color=houses[house]
                            )
                else:
                    x_raw = grades[classes[classe]][house]
                    y_raw = grades[classes[comp_class]][house]
                    
                    pairs = []
                    for x, y in zip(x_raw, y_raw):
                        if x is not None and y is not None:
                            pairs.append((x, y))
                    if not pairs:
                        continue
                    x, y = zip(*pairs)  # it splits a list of (x, y) tuples back into two separate sequences.
                                        # ex: ((1, 2), (3, 4), (5, 6)) -> (1, 3, 5), (2, 4, 6)
                    ax.scatter(x, y, color=houses[house], label=house, alpha = 0.25)

            if index_graph == 13 * 13 - 1:
                ax.legend()
            if (classes[classe] == 12):
                ax.set_xlabel(comp_class[:12] + '.' if len(comp_class) > 8 else comp_class)
            if (classes[comp_class] == 0):
                ax.set_ylabel(classe[:10] + '.' if len(classe) > 8 else classe)
            index_graph += 1


    plt.tight_layout()
    plt.show()

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

    pair_plot(args.dataset)


if __name__ == "__main__":
    main()
