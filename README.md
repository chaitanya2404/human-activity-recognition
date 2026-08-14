# Human Activity Recognition

A classifier that identifies human physical activity (e.g. walking, sitting, standing, laying) from smartphone accelerometer/gyroscope sensor readings, using the classic HAR (Human Activity Recognition) dataset collected from 30 subjects.

## What it does

- Loads the HAR training/test sensor data (561 features + subject ID + activity label)
- Explores class balance, per-subject sample counts, and which raw sensor features separate activities
- Encodes activity labels (fit once on train, reused on test) and shuffles the training set
- Trains an MLP (multi-layer perceptron) neural network classifier (`sklearn.neural_network.MLPClassifier`), comparing SGD and Adam optimizers
- Evaluates both models on the held-out test set: accuracy, per-class precision/recall/F1, and confusion matrices

## Tech stack

- Python, pandas, NumPy
- scikit-learn (`MLPClassifier`, `preprocessing`, `shuffle`, metrics)
- matplotlib, seaborn (visualization)

## Setup & run

Requires Python 3.10+.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset into data/train.csv and data/test.csv
python download_data.py

# 4. Launch Jupyter and run the notebook top to bottom
jupyter notebook human_activity_recognition.ipynb
```

`download_data.py` fetches the dataset directly from the [UCI Machine
Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
(no account or API key needed) and assembles it into the flat
`train.csv` / `test.csv` shape the notebook expects. The dataset itself
(~65 MB) is not checked into this repo — see `.gitignore`.

## Notebook

See [`human_activity_recognition.ipynb`](./human_activity_recognition.ipynb) for the full walkthrough — data loading, EDA, preprocessing, model training, and evaluation, with explanatory markdown throughout.

## Results

Both `MLPClassifier` variants (`hidden_layer_sizes=(90,)`, one hidden layer of 90 units, `random_state=1`) were trained on the 561 sensor features and evaluated on the held-out test subjects (2,947 samples across 9 subjects not seen during training):

| Optimizer | Test accuracy | Iterations to stop | Final training loss |
|---|---|---|---|
| SGD  | 94.74% | 400 (hit the cap, did not converge) | 0.0425 |
| Adam | 94.74% | 186 (converged)                     | 0.0089 |

Both optimizers reached the identical overall test accuracy, but Adam got there in under half the iterations and with a substantially tighter fit to the training data — it's the more efficient and more reliable choice of the two on this problem.

The dominant error mode for both models is confusing **SITTING** and **STANDING** (over 40% of all misclassifications), which tracks with the EDA: those two are the only activity pair where the raw sensor features don't visibly separate, since both are static postures with a similar phone orientation. **LAYING** is classified almost perfectly (99%+ precision/recall) by both models. Full per-class metrics and confusion matrices are in the notebook.
