import numpy as np
import AI_stats_lab as lab


# ============================================================
# Question 1: Confusion Matrix, Metrics, and Threshold Effects
# ============================================================

def test_confusion_matrix_counts_basic():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])

    TP, FP, FN, TN = lab.confusion_matrix_counts(y_true, y_pred)

    assert TP == 1
    assert FP == 1
    assert FN == 1
    assert TN == 1


def test_classification_metrics_basic():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])

    metrics = lab.classification_metrics(y_true, y_pred)

    assert isinstance(metrics, dict)

    assert abs(metrics["recall"] - 0.5) < 1e-6
    assert abs(metrics["fallout"] - 0.5) < 1e-6
    assert abs(metrics["precision"] - 0.5) < 1e-6
    assert abs(metrics["accuracy"] - 0.5) < 1e-6


def test_zero_division_metrics():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])

    metrics = lab.classification_metrics(y_true, y_pred)

    assert metrics["recall"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["fallout"] == 0.0
    assert metrics["accuracy"] == 1.0


def test_apply_threshold_basic():
    scores = np.array([0.2, 0.5, 0.8])
    threshold = 0.5

    preds = lab.apply_threshold(scores, threshold)

    expected = np.array([0, 1, 1])

    assert isinstance(preds, np.ndarray)
    assert np.array_equal(preds, expected)


def test_threshold_metrics_analysis_structure():
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.6, 0.4, 0.1])
    thresholds = np.array([0.0, 0.5, 1.0])

    result = lab.threshold_metrics_analysis(y_true, scores, thresholds)

    assert isinstance(result, list)
    assert len(result) == 3

    required_keys = {
        "threshold",
        "recall",
        "fallout",
        "precision",
        "accuracy"
    }

    for item in result:
        assert isinstance(item, dict)
        assert required_keys.issubset(item.keys())


def test_threshold_metrics_analysis_values():
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.6, 0.4, 0.1])
    thresholds = np.array([0.0, 0.5, 1.0])

    result = lab.threshold_metrics_analysis(y_true, scores, thresholds)

    # threshold = 0.0
    # predictions = [1, 1, 1, 1]
    assert abs(result[0]["recall"] - 1.0) < 1e-6
    assert abs(result[0]["fallout"] - 1.0) < 1e-6

    # threshold = 0.5
    # predictions = [1, 1, 0, 0]
    assert abs(result[1]["recall"] - 1.0) < 1e-6
    assert abs(result[1]["fallout"] - 0.0) < 1e-6

    # threshold = 1.0
    # predictions = [0, 0, 0, 0]
    assert abs(result[2]["recall"] - 0.0) < 1e-6
    assert abs(result[2]["fallout"] - 0.0) < 1e-6


def test_threshold_recall_fallout_trend():
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.6, 0.4, 0.1])
    thresholds = np.array([0.0, 0.5, 1.0])

    result = lab.threshold_metrics_analysis(y_true, scores, thresholds)

    recalls = [item["recall"] for item in result]
    fallouts = [item["fallout"] for item in result]

    # As threshold increases, fewer items are predicted positive.
    assert recalls[0] >= recalls[1] >= recalls[2]
    assert fallouts[0] >= fallouts[1] >= fallouts[2]


# ============================================================
# Question 2: Train Two Classifiers and Evaluate Them
# ============================================================

def test_train_two_classifiers_basic():
    X_train = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 3.0],
    ])

    y_train = np.array([0, 0, 0, 1, 1, 1])

    models = lab.train_two_classifiers(X_train, y_train)

    assert isinstance(models, dict)
    assert "logistic_regression" in models
    assert "decision_tree" in models

    assert hasattr(models["logistic_regression"], "predict")
    assert hasattr(models["logistic_regression"], "predict_proba")
    assert hasattr(models["decision_tree"], "predict")
    assert hasattr(models["decision_tree"], "predict_proba")


def test_evaluate_classifier_basic_answers():
    X_train = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 3.0],
    ])

    y_train = np.array([0, 0, 0, 1, 1, 1])

    X_test = np.array([
        [0.0, 0.5],
        [1.5, 1.5],
    ])

    y_test = np.array([0, 1])

    models = lab.train_two_classifiers(X_train, y_train)

    result = lab.evaluate_classifier(
        models["decision_tree"],
        X_test,
        y_test,
        threshold=0.5
    )

    required_keys = {
        "TP",
        "FP",
        "FN",
        "TN",
        "recall",
        "fallout",
        "precision",
        "accuracy"
    }

    assert isinstance(result, dict)
    assert required_keys.issubset(result.keys())

    assert result["TP"] + result["FP"] + result["FN"] + result["TN"] == len(y_test)

    assert 0 <= result["recall"] <= 1
    assert 0 <= result["fallout"] <= 1
    assert 0 <= result["precision"] <= 1
    assert 0 <= result["accuracy"] <= 1


def test_compare_classifiers_basic():
    X_train = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [2.0, 3.0],
    ])

    y_train = np.array([0, 0, 0, 1, 1, 1])

    X_test = np.array([
        [0.0, 0.5],
        [1.5, 1.5],
    ])

    y_test = np.array([0, 1])

    result = lab.compare_classifiers(
        X_train,
        y_train,
        X_test,
        y_test,
        threshold=0.5
    )

    assert isinstance(result, dict)
    assert "logistic_regression" in result
    assert "decision_tree" in result

    for model_name in ["logistic_regression", "decision_tree"]:
        assert "TP" in result[model_name]
        assert "FP" in result[model_name]
        assert "FN" in result[model_name]
        assert "TN" in result[model_name]
        assert "accuracy" in result[model_name]
        assert "recall" in result[model_name]
        assert "fallout" in result[model_name]
        assert "precision" in result[model_name]
