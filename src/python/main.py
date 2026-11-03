"""Titanic survival prediction with logistic regression."""
 
from pathlib import Path
 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
 
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
 
FEATURES = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "FamilySize", "IsAlone",
    "Embarked_C", "Embarked_Q", "Embarked_S",
    "Title_Master", "Title_Miss", "Title_Mr", "Title_Mrs", "Title_Rare",
]
 
 
def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
 
 
def extract_title(name):
    """Pull the title (Mr, Mrs, Miss...) out of a passenger name."""
    title = name.split(",")[1].split(".")[0].strip()
    mapping = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
    title = mapping.get(title, title)
    return title if title in {"Mr", "Mrs", "Miss", "Master"} else "Rare"
 
 
def prepare(df, fill_values):
    """Apply the same cleaning and feature engineering to any dataset."""
    df = df.copy()
 
    df["Title"] = df["Name"].apply(extract_title)
    print("Extracted Title from Name:", df["Title"].value_counts().to_dict())
 
    for col in ["Age", "Fare", "Embarked"]:
        n_missing = int(df[col].isna().sum())
        df[col] = df[col].fillna(fill_values[col])
        print(f"Filled {n_missing} missing values in {col} with {fill_values[col]}")
 
    df = df.drop(columns=["Cabin", "Ticket", "Name"])
    print("Dropped columns: Cabin (mostly missing), Ticket, Name")
 
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    print("Encoded Sex: male -> 0, female -> 1")
 
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    print("Added FamilySize (SibSp + Parch + 1) and IsAlone (FamilySize == 1)")
 
    df = pd.get_dummies(df, columns=["Embarked", "Title"], dtype=int)
    print("One-hot encoded Embarked and Title")
 
    # Make sure every expected feature column exists (fill missing ones with 0)
    df = df.reindex(columns=list(df.columns.union(FEATURES)), fill_value=0)
    return df
 
 
# ------------------------------------------------------------------
section("Step 1: Load training data")
train = pd.read_csv(DATA_DIR / "train.csv")
print(f"Loaded train.csv: {train.shape[0]} rows, {train.shape[1]} columns")
print("\nFirst 5 rows:")
print(train.head())
 
# ------------------------------------------------------------------
section("Step 2: Explore training data")
print("Missing values per column:")
print(train.isna().sum())
print(f"\nOverall survival rate: {train['Survived'].mean():.3f}")
print("\nSurvival rate by Sex:")
print(train.groupby("Sex")["Survived"].mean().round(3))
print("\nSurvival rate by Pclass:")
print(train.groupby("Pclass")["Survived"].mean().round(3))
 
# ------------------------------------------------------------------
section("Step 3: Clean and engineer features (training data)")
# Fill values come from the TRAINING set only, then reused on the test set
fill_values = {
    "Age": round(float(train["Age"].median()), 2),
    "Fare": round(float(train["Fare"].median()), 2),
    "Embarked": str(train["Embarked"].mode()[0]),
}
print("Fill values computed from training data:", fill_values)
train_clean = prepare(train, fill_values)
print(f"\nFeatures used ({len(FEATURES)}): {FEATURES}")
print("Remaining missing values:", int(train_clean[FEATURES].isna().sum().sum()))
 
# ------------------------------------------------------------------
section("Step 4: Train logistic regression model")
X_train = train_clean[FEATURES]
y_train = train_clean["Survived"]
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
print("Model trained.")
print("\nCoefficients (positive = higher chance of survival):")
coefs = pd.Series(model.coef_[0], index=FEATURES).sort_values(ascending=False)
print(coefs.round(3))
 
# ------------------------------------------------------------------
section("Step 5: Accuracy on training set")
train_acc = accuracy_score(y_train, model.predict(X_train))
print(f"Training accuracy: {train_acc:.4f}")
 
# ------------------------------------------------------------------
section("Step 6: Load test data and predict")
test = pd.read_csv(DATA_DIR / "test.csv")
print(f"Loaded test.csv: {test.shape[0]} rows, {test.shape[1]} columns")
print("Missing values per column:")
print(test.isna().sum())
print()
test_clean = prepare(test, fill_values)
test_clean["Predicted"] = model.predict(test_clean[FEATURES])
print(f"\nPredicted survivors: {int(test_clean['Predicted'].sum())} "
      f"of {len(test_clean)} passengers")
print("\nFirst 10 predictions:")
print(test_clean[["PassengerId", "Predicted"]].head(10).to_string(index=False))
 
# ------------------------------------------------------------------
section("Step 7: Accuracy on test set")
# test.csv has no Survived column. Kaggle's gender_submission.csv is used as
# the reference label (note: it is a baseline that predicts all women survive).
labels = pd.read_csv(DATA_DIR / "gender_submission.csv")
merged = test_clean.merge(labels, on="PassengerId")
test_acc = accuracy_score(merged["Survived"], merged["Predicted"])
print("Compared predictions with gender_submission.csv")
print(f"Test accuracy: {test_acc:.4f}")
 
section("Done")