# Human Activity Recognition

A classifier that identifies human physical activity (e.g. walking, sitting, standing, laying) from smartphone accelerometer/gyroscope sensor readings, using the classic HAR (Human Activity Recognition) dataset collected from 30 subjects.

## What it does

- Loads the HAR training/test sensor data
- Encodes activity labels and shuffles the training set
- Trains an MLP (multi-layer perceptron) neural network classifier (`sklearn.neural_network.MLPClassifier`), comparing SGD and Adam optimizers
- Evaluates classification performance on the held-out test set

## Tech stack

- Python, pandas, NumPy
- scikit-learn (`MLPClassifier`, `preprocessing`, `shuffle`)

## Notebook

See [`human_activity_recognition.ipynb`](./human_activity_recognition.ipynb) for the full walkthrough — data loading, preprocessing, model training, and evaluation.

## Note

This is an early exploratory project (2020) kept here for history. Data loading originally relied on Google Colab + Google Drive; to run it locally, point `pd.read_csv(...)` at a local copy of the HAR dataset instead.
