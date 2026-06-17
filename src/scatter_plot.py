import csv
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "datasets" / "dataset_train.csv"

def scatter_plot(filename: str | Path) -> None:
    
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

    houses = {"Gryffindor": "firebrick", "Slytherin": "green", "Ravenclaw": "royalblue", "Hufflepuff": "gold"}

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

    class_x = "Astronomy"
    idx_x = classes[class_x]
    class_y = "Defense Against the Dark Arts"
    idx_y = classes[class_y]

    fig, ax = plt.subplots()
    for house, color in houses.items():
        x_raw = grades[idx_x][house]
        y_raw = grades[idx_y][house]

        pairs = []
        for x, y in zip(x_raw, y_raw):
            if x is not None and y is not None:
                pairs.append((x, y))
        if not pairs:
            continue
        x, y = zip(*pairs)  # it splits a list of (x, y) tuples back into two separate sequences.
                            # ex: ((1, 2), (3, 4), (5, 6)) -> (1, 3, 5), (2, 4, 6)
        ax.scatter(x, y, color=color, label=house, alpha = 0.25)

    ax.set_xlabel(class_x)
    ax.set_ylabel(class_y)
    ax.set_title(f"{class_x} vs {class_y}")
    ax.legend()
    plt.show()
    return


if __name__ == "__main__":
    scatter_plot(DATASET_PATH)
