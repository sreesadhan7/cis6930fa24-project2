import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


def preprocess_data(input_file):
    """
    Load and preprocess the data, including handling single-instance classes.
    """
    # Load data and add numerical ID
    data = pd.read_csv(input_file, sep="\t", names=["split", "name", "context"], skiprows=1)
    data["id"] = range(1, len(data) + 1)  # Add numerical ID column

    # Split the data into training and validation sets
    train_data = data[data["split"] == "training"].copy()
    val_data = data[data["split"] == "validation"].copy()

    return data, train_data, val_data


def vectorize_data(train_data, val_data):
    """
    Convert context (text) into numerical features using TfidfVectorizer.
    """
    # Use TF-IDF vectorizer with increased features and bi-grams
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 3), stop_words="english")
    X_train = vectorizer.fit_transform(train_data["context"])
    X_val = vectorizer.transform(val_data["context"])
    return X_train, X_val, vectorizer


def train_and_predict(X_train, y_train, X_val):
    """
    Train a RandomForest model and predict redacted names.
    """
    # Train a RandomForest model with optimized hyperparameters
    model = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return model, y_pred


def evaluate_and_save_results(data, val_data, y_true, y_pred, submission_file):
    """
    Evaluate model predictions and save the results to a submission file.
    """
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="micro")
    recall = recall_score(y_true, y_pred, average="micro")
    f1 = f1_score(y_true, y_pred, average="micro")

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")

    # Add predicted names to the validation dataset
    val_data = val_data.copy()
    val_data["predicted_name"] = y_pred

    # Merge predictions back into the original data
    submission_data = data.copy()
    submission_data["predicted_name"] = submission_data["name"]  # Initialize with original names
    submission_data.loc[submission_data["split"] == "validation", "predicted_name"] = submission_data.loc[
        submission_data["split"] == "validation", "id"
    ].map(val_data.set_index("id")["predicted_name"])

    # Merge predictions back into the original dataset
    # submission_data = pd.merge(
    #     data,
    #     val_data[["id", "predicted_name"]],
    #     on="id",
    #     how="left"
    # )

    # # Use predicted names for validation rows
    # submission_data["name"] = submission_data["predicted_name"].fillna(submission_data["name"])

    # Save results to submission file
    submission = submission_data[["id", "name"]]
    submission.to_csv(submission_file, sep="\t", index=False)
    print(f"Submission file saved to {submission_file}")


def main(input_file, submission_file):
    """
    Main function to execute the workflow.
    """
    # Load and preprocess the data
    data, train_data, val_data = preprocess_data(input_file)

    # Vectorize the context text
    X_train, X_val, vectorizer = vectorize_data(train_data, val_data)

    # Encode the target labels
    y_train = train_data["name"]
    y_true = val_data["name"]

    # Train the model and predict on the validation set
    model, y_pred = train_and_predict(X_train, y_train, X_val)

    # Evaluate and save the results
    evaluate_and_save_results(data, val_data, y_true, y_pred, submission_file)


if __name__ == "__main__":
    input_file = "unredactor.tsv"  # Input file containing the data
    submission_file = "submission.tsv"  # Output submission file
    main(input_file, submission_file)