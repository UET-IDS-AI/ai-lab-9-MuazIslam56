"""
AI_stats_lab.py

Lab: Training and Evaluating Classification Models
"""

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# Question 1: Confusion Matrix, Metrics, and Threshold Effects
# ============================================================

def confusion_matrix_counts(y_true, y_pred):
    """
    Compute confusion matrix counts for binary classification.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))

    return (TP, FP, FN, TN)


def classification_metrics(y_true, y_pred):
    """
    Compute classification metrics.
    """

    TP, FP, FN, TN = confusion_matrix_counts(y_true, y_pred)

    recall = TP / (TP + FN) if (TP + FN) != 0 else 0.0
    fallout = FP / (FP + TN) if (FP + TN) != 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0.0
    accuracy = (TP + TN) / (TP + FP + FN + TN)

    return {
        "recall": recall,
        "fallout": fallout,
        "precision": precision,
        "accuracy": accuracy
    }


def apply_threshold(scores, threshold):
    """
    Convert prediction scores into binary predictions.
    """

    scores = np.array(scores)

    return np.where(scores >= threshold, 1, 0)


def threshold_metrics_analysis(y_true, scores, thresholds):
    """
    Analyze how changing threshold affects recall and fallout.
    """

    results = []

    for threshold in thresholds:

        predictions = apply_threshold(scores, threshold)

        metrics = classification_metrics(y_true, predictions)

        results.append({
            "threshold": threshold,
            "recall": metrics["recall"],
            "fallout": metrics["fallout"],
            "precision": metrics["precision"],
            "accuracy": metrics["accuracy"]
        })

    return results


# ============================================================
# Question 2: Train Two Classifiers and Evaluate Them
# ============================================================

def train_two_classifiers(X_train, y_train):
    """
    Train Logistic Regression and Decision Tree classifiers.
    """

    logistic_model = LogisticRegression(max_iter=1000)

    decision_tree_model = DecisionTreeClassifier(random_state=0)

    logistic_model.fit(X_train, y_train)

    decision_tree_model.fit(X_train, y_train)

    return {
        "logistic_regression": logistic_model,
        "decision_tree": decision_tree_model
    }


def evaluate_classifier(model, X_test, y_test, threshold=0.5):
    """
    Evaluate a trained classifier.
    """

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = apply_threshold(probabilities, threshold)

    TP, FP, FN, TN = confusion_matrix_counts(y_test, predictions)

    metrics = classification_metrics(y_test, predictions)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "recall": metrics["recall"],
        "fallout": metrics["fallout"],
        "precision": metrics["precision"],
        "accuracy": metrics["accuracy"]
    }


def compare_classifiers(X_train, y_train, X_test, y_test, threshold=0.5):
    """
    Train and compare two classifiers.
    """

    models = train_two_classifiers(X_train, y_train)

    logistic_results = evaluate_classifier(
        models["logistic_regression"],
        X_test,
        y_test,
        threshold
    )

    decision_tree_results = evaluate_classifier(
        models["decision_tree"],
        X_test,
        y_test,
        threshold
    )

    return {
        "logistic_regression": logistic_results,
        "decision_tree": decision_tree_results
    }


if __name__ == "__main__":
    print("Implement all required functions.")
