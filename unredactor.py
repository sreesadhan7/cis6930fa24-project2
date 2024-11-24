import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import spacy


# Load spaCy for handling interlinked context
nlp = spacy.load("en_core_web_trf")


def preprocess_data(input_file):
    """
    Load and preprocess the data, including splitting into training and validation.
    """
    # Load data and add numerical ID
    data = pd.read_csv(input_file, sep="\t", names=["split", "name", "context"], skiprows=1)
    data["id"] = range(1, len(data) + 1)

    # Split the data into training and validation sets
    train_data = data[data["split"] == "training"].copy()
    val_data = data[data["split"] == "validation"].copy()

    return data, train_data, val_data


def vectorize_data(train_data, val_data, test_data=None):
    """
    Convert context (text) into numerical features using TfidfVectorizer.
    """
    # Use TF-IDF vectorizer with increased features and bi-grams
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), stop_words="english")
    X_train = vectorizer.fit_transform(train_data["context"])
    X_val = vectorizer.transform(val_data["context"])

    # Handle optional test data
    X_test = vectorizer.transform(test_data["context"]) if test_data is not None else None

    return X_train, X_val, vectorizer, X_test


def train_and_predict(X_train, y_train, X_val, X_test=None):
    """
    Train a RandomForest model and predict redacted names for validation and test sets.
    """
    # Training a RandomForest model with optimized hyperparameters
    model = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # Predict on validation data
    y_val_pred = model.predict(X_val)

    # Predict on test data
    y_test_pred = None
    if X_test is not None:
        y_test_pred = model.predict(X_test)

    return model, y_val_pred, y_test_pred


def refine_test_predictions(test_data, model, vectorizer):
    """
    Predict redacted names for test.tsv based on context, using spaCy for refinement.
    """
    refined_predictions = []
    for _, row in test_data.iterrows():
        context = row["context"]

        # Ensure the context is passed as a 2D array to vectorizer and model
        vectorized_context = vectorizer.transform([context])  # Already 2D
        
        # Predict using the model
        predicted_name = model.predict(vectorized_context)[0]  # Get the first (only) prediction

        # Refine prediction with spaCy for interlinked context
        doc = nlp(context)
        spacy_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        refined_name = predicted_name if predicted_name else " ".join(spacy_names)

        refined_predictions.append(refined_name)

    return refined_predictions


def save_test_predictions(test_data, refined_predictions, submission_file):
    """
    Save refined predictions to a submission file.
    """
    test_data = test_data.copy()
    test_data["name"] = refined_predictions  # Added refined predictions to the test data
    test_data[["id", "name"]].to_csv(submission_file, sep="\t", index=False)
    print(f"Test predictions saved to {submission_file}")


def main(train_file, test_file, submission_file):
    """
    Main function to train the model and generate predictions for the test file.
    """
    # Load and preprocess the training data
    data, train_data, val_data = preprocess_data(train_file)

    # Load test data
    test_data = pd.read_csv(test_file, sep="\t", names=["id", "context"])

    # Vectorize the context text
    X_train, X_val, vectorizer, X_test = vectorize_data(train_data, val_data, test_data)

    # Encode the target labels
    y_train = train_data["name"]
    y_true = val_data["name"]

    # Train the model and predict on validation and test sets
    model, y_val_pred, y_test_pred = train_and_predict(X_train, y_train, X_val, X_test)

    # Evaluate the model on the validation set
    print("\nValidation Metrics:")
    print(f"Accuracy: {accuracy_score(y_true, y_val_pred):.2f}")
    print(f"Precision: {precision_score(y_true, y_val_pred, average='micro'):.2f}")
    print(f"Recall: {recall_score(y_true, y_val_pred, average='micro'):.2f}")
    print(f"F1 Score: {f1_score(y_true, y_val_pred, average='micro'):.2f}")

    # Generate refined predictions for test data
    refined_predictions = refine_test_predictions(test_data, model, vectorizer)

    # Saving refined predictions for the test set
    save_test_predictions(test_data, refined_predictions, submission_file)


if __name__ == "__main__":
    train_file = "unredactor.tsv"  # Training file
    test_file = "test.tsv"  # Test file containing id and context columns
    submission_file = "submission.tsv"  # Output submission file
    main(train_file, test_file, submission_file)