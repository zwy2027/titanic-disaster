# Titanic Disaster: Survival Prediction

Northwestern University MLDS · Introduction to Data Engineering · Homework 3

This repository trains a **logistic regression** model that predicts whether a passenger survived the Titanic disaster (April 15, 1912). The same pipeline is implemented twice, each in its own Docker container:

- **Python** (pandas + scikit-learn), in `src/python/`
- **R** (base R `glm` + dplyr), in `src/r/`

Each script loads the data, explores it, cleans it, engineers features, trains the model, and prints the accuracy on the training set and the test set. Every step prints its changes to the terminal, so you only need to read the output.

---

## Repository structure

```
titanic-disaster/
├── .gitignore
├── CODEOWNERS
├── README.md
└── src/
    ├── data/                    ← you put the dataset here (not included in the repo)
    ├── python/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── main.py
    └── r/
        ├── Dockerfile
        ├── install_packages.R
        └── main.R
```

---

## Prerequisites

You only need two things installed. You do **not** need Python or R on your machine; Docker provides them.

1. **Git**: check with `git --version`
2. **Docker Desktop**: download from https://www.docker.com/products/docker-desktop/ , open it, and wait until it shows **Engine running**. Check with `docker --version`

---

## Step 1: Clone the repository

```bash
git clone https://github.com/zwy2027/titanic-disaster.git
cd titanic-disaster
```

All commands below are run from this `titanic-disaster` folder.

---

## Step 2: Download the data

The dataset is not included in this repository. Download it from Kaggle:

1. Go to https://www.kaggle.com/competitions/titanic/data (a free Kaggle account is required; click **Join Competition / I Understand and Accept** if asked).
2. Download these three files:
   - `train.csv`
   - `test.csv`
   - `gender_submission.csv`
3. Create the data folder and move the three files into it:

```bash
mkdir -p src/data
mv ~/Downloads/train.csv ~/Downloads/test.csv ~/Downloads/gender_submission.csv src/data/
```

(If your browser saved the files somewhere else, or downloaded a zip file, unzip it and move the three `.csv` files into `src/data/` manually.)

4. Check that the files are in place:

```bash
ls src/data
```

You should see:

```
gender_submission.csv  test.csv  train.csv
```

---

## Step 3: Run the Python container

```bash
docker build -t titanic-python src/python
docker run --rm -v "$(pwd)/src/data:/data" titanic-python
```

- `docker build` builds the image from `src/python/Dockerfile` and installs the pinned packages in `requirements.txt` (pandas 3.1.0, numpy 2.5.4, scikit-learn 1.9.1) on Python 3.12.
- `docker run` starts the container. `-v "$(pwd)/src/data:/data"` mounts your local `src/data` folder into the container at `/data`, so the script can read the CSV files. `--rm` removes the container after it finishes.

The first build takes a few minutes while the base image and packages download.

---

## Step 4: Run the R container

```bash
docker build -t titanic-r src/r
docker run --rm -v "$(pwd)/src/data:/data" titanic-r
```

- The image is based on `rocker/r-ver:4.6.1`, which pins CRAN to a dated snapshot (2026-10-28), so the same package versions are installed on every build.
- `install_packages.R` installs `dplyr` during the build and prints the installed version.

The first build takes a few minutes.

---

## What the output shows

Both containers print the same seven sections:

| Section | What happens |
|---|---|
| Step 1: Load training data | Reads `train.csv`, prints its shape and first rows |
| Step 2: Explore training data | Missing values per column; survival rate overall, by Sex, and by Pclass |
| Step 3: Clean and engineer features | Fills missing Age and Fare with the training median and Embarked with the most common port; drops Cabin, Ticket, and Name; encodes Sex; creates Title (from Name), FamilySize, and IsAlone; encodes Embarked and Title |
| Step 4: Train logistic regression | Fits the model and prints its coefficients |
| Step 5: Accuracy on training set | Accuracy of the model on `train.csv` |
| Step 6: Load test data and predict | Applies the same cleaning to `test.csv` and predicts survival |
| Step 7: Accuracy on test set | Accuracy of the predictions on the test set |

Fill values (medians and the most common port) are computed from the **training set only** and reused on the test set, so no information from the test set leaks into the model.

Both models reach a training accuracy of about **0.83** (Python: 0.8339, R: 0.8328).

**Note on test accuracy:** Kaggle's `test.csv` does not include the true `Survived` labels. The scripts therefore compare their predictions with `gender_submission.csv`, Kaggle's sample file, which predicts that every woman survived and every man died. The "test accuracy" is the agreement between our model and that baseline, not a true accuracy.

**Small difference between versions:** the R model does not use `FamilySize` as a feature, because it equals `SibSp + Parch + 1` and `glm` cannot estimate a coefficient for an exactly collinear variable. The Python model keeps it, since scikit-learn's regularized logistic regression handles it.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `command not found: docker` | Install Docker Desktop, open it, then open a new terminal |
| `Cannot connect to the Docker daemon` | Open Docker Desktop and wait for **Engine running** |
| `FileNotFoundError` / `cannot open file '/data/train.csv'` | Make sure the three CSV files are in `src/data/` and that you run the commands from the `titanic-disaster` folder |
| Windows PowerShell | Replace `$(pwd)` with `${PWD}`, e.g. `docker run --rm -v "${PWD}/src/data:/data" titanic-python` |

---

## Optional: run without Docker

For development only; graders do not need this.

**Python** (requires Python 3.12+):

```bash
cd src/python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

**R** (requires R with dplyr installed):

```bash
cd src/r
Rscript main.R
```

When run locally, each script reads the data from `../data` (that is, `src/data`).
