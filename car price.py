import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "dataset"


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Make sure the repository contains the 'dataset' file."
        )

    data = pd.read_csv(DATA_PATH)

    print(data.info())
    print(data.describe())
    print(data.head())

    print(data.isnull().sum())
    data = data.ffill()

    data = pd.get_dummies(data, drop_first=True)

    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    if "Price" not in data.columns:
        raise ValueError("Dataset must contain a 'Price' target column.")

    X = data.drop("Price", axis=1)
    y = data["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")


if __name__ == "__main__":
    main()
