# cis6930fa24 -- Project2

Name: Sai Sree Sadhan Polimera

# Assignment Description 

This project focuses on building an Unredactor application using Python, which automates the recovery of redacted names in textual data. The application is designed to process training and test datasets, learning patterns from context to accurately predict and replace redacted names. The system leverages machine learning techniques such as `TF-IDF vectorization` and classification model: `Random Forest` for context-based predictions. Additionally, spaCy’s Named Entity Recognition (NER) capabilities are incorporated to refine predictions. This tool is essential for tasks requiring automated restoration of sensitive data, providing a scalable and modular solution for textual data processing.

### Assignment Objective:

The primary objectives of this assignment are:
1) Data Preparation and Model Training:
    - Preprocess a provided dataset (unredactor.tsv) containing redacted names and their surrounding contexts.
    - Train a machine learning model using the training subset to predict names based on contextual information.
2) Prediction and Refinement:
    - Predict redacted names in the validation dataset and evaluate performance using metrics such as accuracy, precision, recall, and F1-score.
    - Use spaCy to refine predictions with Named Entity Recognition (NER) for improved context-aware name predictions.
3) Final Submission:
    - Run the trained model on a separate test dataset (test.tsv) and generate a prediction file (submission.tsv) in the required format for evaluation.
   
### Requirements:
1) Dataset:
    - Input Files:
        - unredactor.tsv: Contains training and validation examples with columns—split, name, and context.
        - test.tsv: Contains columns—id and context, where names need to be predicted.
    - Output File:
        - submission.tsv: Contains columns—id and name with predicted names for the test data.
2) Feature Extraction:
    - Use TF-IDF Vectorization with increased feature count (up to 10,000) and support for n-grams (up to trigrams).
3) Model:
    - Train a Random Forest Classifier to predict redacted names based on context.
    - Ensure model parameters (e.g., n_estimators, max_depth) are optimized for performance.
4) NER Integration:
    - Incorporate spaCy’s en_core_web_trf model for Named Entity Recognition (NER) to refine predictions for names in complex contexts.
5) Evaluation Metrics:
    - Compute and report:
        - Accuracy
        - Precision
        - Recall
        - F1-score
6) Code Organization:
    - Modular functions for:
        - Data preprocessing
        - Feature vectorization
        - Model training and prediction
        - Refinement using spaCy
7) Final Submission:
    - Test the trained model on test.tsv and generate a submission file (submission.tsv) in tab-separated format, including columns:
        - id: Test sample ID.
        - name: Predicted name for the redacted entity.

# How to install
Install pipenv using the command: 

      pip install pipenv

Install spacy library using the command: 

      pipenv install spacy

Install en_core_web_trf pipeline using the command: 

      pipenv run python -m spacy download en_core_web_trf

Install scikit-learn library using the command:

      pipenv install scikit-learn

Install pytest testing framework using the command: 

      pipenv install pytest 

## How to run
To execute the project, navigate to the project directory and run the following commands:

1) To get the output under the files directory, use command:

        python unredactor.py

        (or)

        pipenv run python unredactor.py

## Test Cases Run
Running the following pipenv command runs the pytest cases. This project have 1 test cases.
command to run test cases: 

      pipenv run python -m pytest -v

## Functions
### unredactor.py

1) **main(train_file, test_file, submission_file)**
    
    Main function to train the model and generate predictions for the test file.

    This function orchestrates the workflow by:
    - Loading and preprocessing the training data (unredactor.tsv).
    - Training and evaluating the model on the validation set and printing metrics (Accuracy, Percision, Recall, F1 Score) based on validation data.
    - Generating refined predictions for the test dataset (test.tsv).
    - Saving the predictions to a submission file.

    Args:
    - train_file (str): Path to the training file (unredactor.tsv).
    - test_file (str): Path to the test file (test.tsv).
    - submission_file (str): Path to the output submission file (submission.tsv).

2) **preprocess_data(input_file)**

    Load and preprocess the data from the specified file.

    This function reads the dataset, assigns a numerical ID to each row, and splits the data into training and validation sets based on the 'split' column.

    Args:
    - input_file (str): Path to the input dataset file (unredactor.tsv).

    Returns:
    - tuple: A tuple containing:
        - data (pd.DataFrame): The complete dataset with assigned IDs.
        - train_data (pd.DataFrame): Subset of data for training (split == "training").
        - val_data (pd.DataFrame): Subset of data for validation (split == "validation").

3) **vectorize_data(train_data, val_data, test_data=None)**

    Convert textual data (context column) into numerical features using TF-IDF vectorization.

    This function transforms the context column of the training, validation, and optionally test datasets into numerical feature representations.

    Args:
    - train_data (pd.DataFrame): Training dataset containing the context column.
    - val_data (pd.DataFrame): Validation dataset containing the context column.
    - test_data (pd.DataFrame, optional): Test dataset containing the context column.

    Returns:
    - tuple: A tuple containing:
        - X_train (sparse matrix): TF-IDF features for the training data.
        - X_val (sparse matrix): TF-IDF features for the validation data.
        - vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
        - X_test (sparse matrix, optional): TF-IDF features for the test data (if test_data is provided).

4) **train_and_predict(X_train, y_train, X_val, X_test=None)**

    Train a Random Forest model and predict redacted names for validation and test sets.

    This function trains a Random Forest classifier on the training data and makes predictions for the validation and optionally the test datasets.

    Args:
    - X_train (sparse matrix): TF-IDF features for the training data.
    - y_train (pd.Series): Labels (names) for the training data.
    - X_val (sparse matrix): TF-IDF features for the validation data.
    - X_test (sparse matrix, optional): TF-IDF features for the test data (if provided).

    Returns:
    - tuple: A tuple containing:
        - model (RandomForestClassifier): The trained Random Forest model.
        - y_val_pred (np.ndarray): Predictions for the validation set.
        - y_test_pred (np.ndarray, optional): Predictions for the test set (if X_test is provided).

5) **refine_test_predictions(test_data, model, vectorizer)**

    Refine predictions for the test dataset by combining model predictions and spaCy NER.

    This function uses the trained model to predict redacted names for the test dataset and refines these predictions by extracting named entities using spaCy for interlinked contexts.

    Args:
    - test_data (pd.DataFrame): Test dataset containing the context column.
    - model (RandomForestClassifier): The trained Random Forest model.
    - vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.

    Returns:
    - list: A list of refined predicted names for the test dataset.

6) **save_test_predictions(test_data, refined_predictions, submission_file)**

    Save refined predictions for the test dataset to a submission file.

    This function writes the predicted names along with their corresponding IDs to a TSV file in the required format.

    Args:
    - test_data (pd.DataFrame): Test dataset containing the ID column.
    - refined_predictions (list): Refined predicted names for the test dataset.
    - submission_file (str): Path to the output submission file.

### test_functions.py
1) ** **
    - 

## Model Performance and Analysis

### Observed Results and Analysis:

The model achieved low evaluation scores across all metrics (accuracy, precision, recall, and F1-score were approximately 0.04).

![image](https://github.com/user-attachments/assets/3502c4a1-36c3-4442-bdff-d725fa6da14e)

While these results are below expectations, they highlight the inherent complexity of the task and several challenges that impacted the model's performance:
- The redacted names often lack sufficient context, making it difficult for the model to accurately predict them.
- The Random Forest Classifier and TF-IDF vectorization, though robust, were not sufficient to capture the nuanced semantic patterns required for this task.
- spaCy's Named Entity Recognition (NER) helped refine predictions but was constrained by the limitations of the dataset.

### Challenges Impacting Model Performance:
1) Ambiguity in Contextual Information
    - The provided contexts often lack sufficient detail to uniquely identify the redacted names, making accurate predictions difficult.
2) Data Limitations
    - The training dataset may be too small or not diverse enough to allow the model to generalize effectively.
    - The assumption that the redacted block's length matches the name's length does not always hold, leading to prediction mismatches.
3) Model and Feature Extraction Limitations
    - The Random Forest Classifier, while robust, may not capture the nuances in textual data where semantic meaning is crucial.
    - TF-IDF vectorization focuses on frequency-based features but does not fully capture contextual relationships or semantic meaning between words.
4) Task Complexity
    - Recovering redacted names is inherently challenging, as it often requires external knowledge or broader contextual understanding, which is not present in the dataset.

### Efforts and Experimentation:
1) Parameter Tuning:
    - Several hyperparameter combinations (e.g: max_features, ngram_range, n_estimators) were tested to improve performance.
2) Feature Engineering:
    - Advanced feature extraction techniques, such as n-grams and removing stop words, were incorporated to capture contextual patterns.
3) Refinement with spaCy:
    - The spaCy NER model was integrated to refine predictions, especially for contexts containing named entities. However, this improvement was limited by the quality of the dataset.

## Bugs and Assumptions
1) Handling Empty or Incorrect Contexts: The function refine_test_predictions assumes every context will contain valid information. If the context is empty or poorly formatted, it might result in incorrect predictions or crashes.
2) Single Redaction Per Context: Assumes that each context contains only one redacted name to predict. Multiple redactions in a single context might not be handled.
3) Assumption: Assumes that TF-IDF vectorization is enough to capture the nuances in the text for predicting names. However, semantic meaning (e.g., relationships between words) might not be fully captured.
4) Ambiguity in Context: Many contexts in the dataset are highly ambiguous and do not provide sufficient information to accurately predict the redacted name. For example, generic sentences without specific references can confuse the model.
5) Assumption: The surrounding text in the context column provides enough information to uniquely identify the redacted name. This can create ambiguity or generic contexts can reduce prediction accuracy.