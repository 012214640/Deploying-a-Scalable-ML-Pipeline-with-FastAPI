import pytest
import numpy as np
from ml.model import compute_model_metrics, train_model, inference

def test_train_model():
    """
    # Test that model has predict method and is able to generate predictions as intended
    """
    X = np.array([[0,1],[1,0],[1,1],[0,0]])
    y = np.array([0,1,1,0])

    model = train_model(X, y)

    assert hasattr(model, "predict")

    preds = model.predict(X)
    assert len(preds) == len(y)

    pass

def test_compute_model():
    """
    # Verify that compute model metrics are within the expected value range
    """
    y = np.array([1,0,1,1])
    preds = np.array([1,0,0,1])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision >= 0 and precision <= 1
    assert recall >= 0 and recall <= 1
    assert fbeta >= 0 and fbeta <= 1

    pass

def test_training_reproducibility():
    """
    # Ensure that random state gives the same results for different runs of the training model
    """
    X = [[0], [1], [0], [1]]
    y = [0, 1, 0, 1]

    model1 = train_model (X, y)
    model2 = train_model (X, y)

    pred1 = inference(model1, X)
    pred2 = inference(model2, X)

    assert (pred1 == pred2).all()

    pass
