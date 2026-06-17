import csv
from pathlib import Path
from lib import list_std, list_mean, min_idx
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "datasets" / "dataset_train.csv"

def histogram(filename: str | Path) -> None:
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

    grades = [{"Gryffindor": [], "Slytherin": [], "Ravenclaw": [], "Hufflepuff": []} for _ in classes]

    with open(filename, "r") as file:
        data = csv.DictReader(file)
        for row in data:
            for classe in classes:
                if row[classe]:
                    grades[classes[classe]][row["Hogwarts House"]].append(float(row[classe]))

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))

    std_of_mean_by_houses = [0.0] * len(classes)

    for classe in classes:
        mean_of_houses = [ ]
        for house in houses:
            mean_of_houses.append(list_mean(grades[classes[classe]][house]))
        std_of_mean_by_houses[classes[classe]] = list_std(mean_of_houses)

    for index, (ax, classe) in enumerate(zip(axes.flat, classes)):
        for house in houses:
            ax.hist(
                grades[classes[classe]][house],
                bins=20,
                alpha=0.5,
                label=house,
                color=houses[house]
            )
        if index == min_idx(std_of_mean_by_houses):
            ax.set_facecolor("#fff3cd")  # fond jaune clair
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)

        ax.set_title(classe)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        if classe == "Flying":
            ax.legend()


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    histogram(DATASET_PATH)
