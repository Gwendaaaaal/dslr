# DSLR

Data science project recreating the Hogwarts Sorting Hat using a
one-vs-all logistic regression model trained with gradient descent.

## Requirements

- Python 3.10 or later

## Installation

```bash
git clone https://github.com/Gwendaaaaal/dslr.git
cd dslr

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Data analysis

```bash
python3 src/describe.py datasets/dataset_train.csv
```

## Data visualization

```bash
python3 src/histogram.py datasets/dataset_train.csv
python3 src/scatter_plot.py datasets/dataset_train.csv
python3 src/pair_plot.py datasets/dataset_train.csv
```
