# Iris Flower Classifier (Python + Machine Learning)

A machine learning example prepared for me by my AI assistant: a small
program that learns to recognize iris flower species from their petal and
sepal measurements.

- **Dataset:** the classic Iris dataset (150 flowers, 3 species), included with scikit-learn
- **Algorithm:** K-Nearest Neighbors (KNN)
- **Measured result:** 100% accuracy on this repo's test split (30 flowers, random_state=42). This is one measurement on a small, easy dataset, not a general accuracy claim

## Setup

You need Python 3 and scikit-learn:

```bash
pip install scikit-learn
```

## Run

```bash
python iris_classifier.py
```

Example output:

```
Dataset: 150 flowers, 4 measurements each
Accuracy on unseen flowers: 100%
...
Example prediction for [5.1, 3.5, 1.4, 0.2]: setosa
```

## Concepts to study in this project

- Splitting data into training and test sets (and why you must not test on training data)
- Training a classifier with scikit-learn's `fit` / `predict` API
- Reading a classification report (precision, recall, f1-score)

## Notes

A learning example prepared for me by my AI assistant to study and extend while
I take the Machine Learning course at the Faculty of Computers and Artificial
Intelligence, Minya University.
