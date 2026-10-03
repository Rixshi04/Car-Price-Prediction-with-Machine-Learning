# Car Price Prediction with Machine Learning

## Overview
This project uses machine learning to predict car prices from features such as age, mileage, horsepower, and brand. It includes data preprocessing, exploratory visualization, one-hot encoding, Linear Regression training, and evaluation.

## Project structure

- `car price.py` — main training and evaluation script
- `dataset` — bundled tab-separated dataset used by the script
- `requirements.txt` — Python dependencies

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/Rixshi04/Car-Price-Prediction-with-Machine-Learning.git
cd Car-Price-Prediction-with-Machine-Learning
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run from the repository root

Use:

```bash
python "car price.py"
```

The script resolves the bundled `dataset` file relative to the script location, so it works when launched from the repository root.

The script will:
- Load and inspect the dataset
- Check and forward-fill missing values
- One-hot encode the `Brand` column
- Display a correlation heatmap
- Train a Linear Regression model
- Report Mean Squared Error and R-squared on the test set

## Dataset

The repository includes the dataset directly as `dataset`. Its columns are:

- `Age`
- `Mileage`
- `Horsepower`
- `Brand`
- `Price`

## Notes

The project is a simple educational regression example. The reported metrics depend on the bundled dataset and the fixed train/test split.

## License

See the repository license file.
