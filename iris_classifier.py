"""Iris flower classifier.

A first machine learning project: train a small model to recognize iris
flower species from petal and sepal measurements.

Dataset: the classic Iris dataset (150 flowers, 3 species), built into scikit-learn.
Model: K-Nearest Neighbors, a simple and easy-to-understand algorithm.
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


def main():
    # 1. Load the dataset (150 flowers, 4 measurements each, 3 species).
    iris = load_iris()
    X = iris.data        # sepal length, sepal width, petal length, petal width
    y = iris.target      # 0 = setosa, 1 = versicolor, 2 = virginica

    print(f"Dataset: {X.shape[0]} flowers, {X.shape[1]} measurements each")

    # 2. Split into training data (80%) and test data (20%).
    # The model learns from the training part only; the test part checks
    # how well it handles flowers it has never seen.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train the model: KNN classifies a flower by looking at the
    # 3 most similar flowers it already knows.
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)

    # 4. Evaluate on the test data.
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy on unseen flowers: {accuracy:.0%}")
    print()
    print(classification_report(y_test, predictions, target_names=iris.target_names))

    # 5. Try the model on a new flower.
    new_flower = [[5.1, 3.5, 1.4, 0.2]]  # sepal length/width, petal length/width
    species = model.predict(new_flower)[0]
    print(f"Example prediction for {new_flower[0]}: {iris.target_names[species]}")


if __name__ == "__main__":
    main()
