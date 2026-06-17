import csv
from pathlib import Path

from lib import first_quartile, list_std, list_sum, median, third_quartile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "datasets" / "dataset_train.csv"

def describe(filename: str | Path) -> None:
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

    count_list = [0] * len(classes)
    mean_list = [0.0] * len(classes)
    std_list = [0.0] * len(classes)
    min_list = [0.0] * len(classes)
    firstquart_list = [0.0] * len(classes)
    median_list = [0.0] * len(classes)
    thirdquart_list = [0.0] * len(classes)
    max_list = [0.0] * len(classes)
    grades_list = [[] for _ in classes]

    with open(filename, "r") as file:
        data = csv.DictReader(file)

        for row in data:
            for classe in classes:
                if row[classe]:
                    count_list[classes[classe]] += 1
                    grades_list[classes[classe]].append(float(row[classe]))

    for i in range(len(grades_list)):
        mean_list[i] = list_sum(grades_list[i]) / count_list[i]
        std_list[i] = list_std(grades_list[i], mean_list[i])
        grades_list[i].sort()
        min_list[i] = grades_list[i][0]
        firstquart_list[i] = first_quartile(grades_list[i])
        median_list[i] = median(grades_list[i])
        thirdquart_list[i] = third_quartile(grades_list[i])
        max_list[i] = grades_list[i][-1]

    COL_W = 30

    print("".ljust(COL_W), end="")
    for classe in classes:
        print(classe.ljust(COL_W), end="")

    rows = [
        ("Count", count_list, ".2f"),
        ("Mean", mean_list, ".6f"),
        ("Std", std_list, ".6f"),
        ("Min", min_list, ".6f"),
        ("25%", firstquart_list, ".6f"),
        ("50%", median_list, ".6f"),
        ("75%", thirdquart_list, ".6f"),
        ("Max", max_list, ".6f"),
    ]

    for label, data, fmt in rows:
        print(f"\n{label.ljust(COL_W)}", end="")
        for val in data:
            formatted = format(val, fmt)
            print(formatted.ljust(COL_W), end="")

    print("")


if __name__ == "__main__":
    describe(DATASET_PATH)
