from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

print("Container is running!")
print("pandas version:", pd.__version__)

train = pd.read_csv(DATA_DIR / "train.csv")
print("Loaded train.csv with shape:", train.shape)