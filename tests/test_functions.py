import os
import pytest
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from unredactor import (
    preprocess_data,
    vectorize_data,
    train_and_predict,
    refine_test_predictions,
    save_test_predictions
)

# Sample Data
TRAIN_DATA = """split\tname\tcontext
training\tJohn Doe\tJohn went to the market to buy groceries.
training\tJane Doe\tJane is a software engineer at a tech company.
validation\tEmily Clark\tEmily loves painting and often visits art galleries.
"""
TEST_DATA = """id\tcontext
1\tJohn visited his friend at the hospital.
2\tJane presented her project at the tech conference.
"""

# Paths for temporary test files
TRAIN_FILE = "test_train.tsv"
TEST_FILE = "test_test.tsv"
SUBMISSION_FILE = "test_submission.tsv"

@pytest.fixture
def setup_data():
    """
    Fixture to create temporary train and test files for testing.
    """
    with open(TRAIN_FILE, "w") as f:
        f.write(TRAIN_DATA)
    with open(TEST_FILE, "w") as f:
        f.write(TEST_DATA)

    yield  # The test runs here

    # Cleanup after test
    if os.path.exists(TRAIN_FILE):
        os.remove(TRAIN_FILE)
    if os.path.exists(TEST_FILE):
        os.remove(TEST_FILE)
    if os.path.exists(SUBMISSION_FILE):
        os.remove(SUBMISSION_FILE)


def test_preprocess_data(setup_data):
    """
    Test preprocess_data function to ensure correct splitting and ID assignment.
    """
    data, train_data, val_data = preprocess_data(TRAIN_FILE)
    
    # Check total rows and columns
    assert data.shape[0] == 3
    assert "id" in data.columns

    # Check training and validation split
    assert train_data.shape[0] == 2
    assert val_data.shape[0] == 1


def test_vectorize_data(setup_data):
    """
    Test vectorize_data function to ensure proper TF-IDF vectorization.
    """
    data, train_data, val_data = preprocess_data(TRAIN_FILE)
    X_train, X_val, vectorizer, _ = vectorize_data(train_data, val_data)

    # Check vectorized dimensions
    assert X_train.shape[0] == 2
    assert X_val.shape[0] == 1
    assert isinstance(vectorizer, TfidfVectorizer)


def test_train_and_predict(setup_data):
    """
    Test train_and_predict function to ensure the model is trained and predictions are made.
    """
    data, train_data, val_data = preprocess_data(TRAIN_FILE)
    X_train, X_val, _, _ = vectorize_data(train_data, val_data)
    y_train = train_data["name"]
    y_val = val_data["name"]

    model, y_val_pred, _ = train_and_predict(X_train, y_train, X_val)

    # Check predictions and model
    assert len(y_val_pred) == len(y_val)
    assert isinstance(model, RandomForestClassifier)


def test_save_test_predictions(setup_data):
    """
    Test save_test_predictions to ensure predictions are saved correctly.
    """
    test_data = pd.read_csv(TEST_FILE, sep="\t")
    refined_predictions = ["John Doe", "Jane Doe"]

    save_test_predictions(test_data, refined_predictions, SUBMISSION_FILE)

    # Check if the submission file is created
    assert os.path.exists(SUBMISSION_FILE)

    # Validate the contents of the submission file
    submission_data = pd.read_csv(SUBMISSION_FILE, sep="\t")
    assert submission_data.shape[0] == 2
    assert "id" in submission_data.columns
    assert "name" in submission_data.columns
    assert list(submission_data["name"]) == refined_predictions