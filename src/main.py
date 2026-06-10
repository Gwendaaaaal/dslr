import csv
from .lib import *

def main():
    obj = jsp("../datasets/dataset_train.csv")
    obj.describe()
    return

#TODO: remove class ?
class jsp:
    def __init__(self, filename: str) :
        self.file = open(filename, "r")
        self.data = csv.DictReader(self.file)

    def __del__(self) :
        self.file.close()

    def describe(self) :
        self.classes = {
            "Arithmancy": 0,
            "Astronomy": 1,
            "Herbology": 2,
            "Defense Against the Dark Arts": 3,
            "Divination" : 4,
            "Muggle Studies" : 5,
            "Ancient Runes" : 6,
            "History of Magic" : 7,
            "Transfiguration" : 8,
            "Potions" : 9,
            "Care of Magical Creatures" : 10,
            "Charms" : 11,
            "Flying" : 12
        }
        self.count_list = [0,0,0,0,0,0,0,0,0,0,0,0,0]        # num elements
        self.mean_list = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.std_list = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.min_list = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.firstquart_list = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.median_list = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.thirdquart_list = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        self.max_list = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]

        self.grades_list = [[],[],[],[],[],[],[],[],[],[],[],[],[]]

        for row in self.data:
            for classe in self.classes.keys():
                if row[classe]:
                    # Count
                    self.count_list[self.classes[classe]] += 1
                    # insert into list
                    self.grades_list[self.classes[classe]].append(float(row[classe])) 

        for i in range(len(self.grades_list)):
            self.mean_list[i] = list_sum(self.grades_list[i]) / self.count_list[i]
            self.std_list[i] = list_std(self.grades_list[i], self.mean_list[i])
            self.grades_list[i].sort()
            self.min_list[i] = self.grades_list[i][0]
            self.firstquart_list[i] = first_quartile(self.grades_list[i])
            self.median_list[i] = median(self.grades_list[i])
            self.thirdquart_list[i] = third_quartile(self.grades_list[i])
            self.max_list[i] = self.grades_list[i][-1]

        # print part
        COL_W = 30  # largeur de chaque colonne

        # Header
        print("".ljust(COL_W), end="")
        for classe in self.classes.keys():
            print(classe.ljust(COL_W), end="")

        rows = [
            ("Count",  self.count_list,      ".2f"),
            ("Mean",   self.mean_list,       ".6f"),
            ("Std",    self.std_list,        ".6f"),
            ("Min",    self.min_list,        ".6f"),
            ("25%",    self.firstquart_list, ".6f"),
            ("50%",    self.median_list,     ".6f"),
            ("75%",    self.thirdquart_list, ".6f"),
            ("Max",    self.max_list,        ".6f"),
        ]

        for label, data, fmt in rows:
            print(f"\n{label.ljust(COL_W)}", end="")
            for val in data:
                formatted = format(val, fmt)
                print(formatted.ljust(COL_W), end="")

        print("")


if __name__ == "__main__":
    main()
