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
Running the following pipenv command runs the pytest cases. This project have 4 test cases.
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
1) **setup_data()**
    - Fixture to set up temporary train and test files for testing. Describes the creation of temporary files for testing and their cleanup after the test.
    
    - Creates temporary files with sample data for training and testing, and ensures cleanup after tests are executed.

    Returns:
    - None: The test runs after this fixture setup.

2) **test_preprocess_data(setup_data)**
    - Test the `preprocess_data` function to ensure correct data loading, splitting into training and validation sets, and ID assignment.

    - Validates:
        - The total number of rows and columns in the data.
        - The presence of the 'id' column in the dataset.
        - The correct number of training and validation examples.

3) **test_vectorize_data(setup_data)**
    - Test the `vectorize_data` function to ensure proper conversion of text data into numerical features using TF-IDF vectorization.

    - Validates:
        - The correct number of training and validation samples.
        - The use of the `TfidfVectorizer` for text vectorization.
        - The dimensions of the output TF-IDF matrices.

4) **test_train_and_predict(setup_data)**
    - Test the `train_and_predict` function to ensure that:
        - The RandomForest model is trained on the training data.
        - Predictions are generated for the validation data.

    - Validates:
        - The length of the predictions matches the validation labels.
        - The returned model is an instance of `RandomForestClassifier`.

5) **test_save_test_predictions(setup_data)**
    - Test the `save_test_predictions` function to ensure that:
        - The predicted names are correctly saved to the submission file.
        - The output file has the expected format and content.

    - Validates:
        - The creation of the `submission.tsv` file.
        - The presence of the required columns ('id', 'name') in the output.
        - The accuracy of the predicted names written to the file.

https://github.com/user-attachments/assets/416c2e41-7e9a-447f-b85e-216d6e943bcc

## Model Performance and Analysis

### Observed Results and Analysis:

The model achieved low evaluation scores across all metrics (accuracy, precision, recall, and F1-score were approximately 0.04).

`RandomForestClassifier` and `TfidfVectorizer` are used in your code:

1. `RandomForestClassifier`:
- Robustness and Non-Linearity: Random Forest is an ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting. It works well with high-dimensional data and can handle non-linear relationships in the data effectively.
- Feature Importance: Random Forest provides insights into which features (words or n-grams from the TF-IDF vectorizer) contribute the most to the predictions, which is valuable in understanding model behavior.
- Versatility and Scalability: Random Forest is versatile and performs well on a wide range of tasks, including text classification. Its parallelism and ability to handle large datasets make it suitable for text-based applications.

2. `TfidfVectorizer`:
- Capturing Importance of Words: TF-IDF (Term Frequency-Inverse Document Frequency) assigns weights to words based on their frequency in a document and across the entire dataset. It helps highlight important words while reducing the influence of common, less meaningful words like "the" or "and."
- Handling Sparse Data: Text data is inherently sparse, with many unique words in the dataset. TF-IDF converts text into numerical features suitable for machine learning algorithms like Random Forest, preserving meaningful relationships while keeping the feature space manageable.
- Customizable Features: TF-IDF allows customization of n-grams, stopword removal, and the maximum number of features, enabling it to capture meaningful word sequences (e.g., bi-grams, tri-grams) and focus on the most relevant features for the task.

By combining `RandomForestClassifier` with `TfidfVectorizer`, your code creates a pipeline that efficiently transforms raw text data into numerical representations (using TF-IDF) and leverages the robustness of Random Forest to perform accurate and interpretable text classification.

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

## Challenges and Limitations
Despite the efforts to create an accurate and efficient model, there are some inherent challenges and limitations that impact the output quality and the resulting metric scores:

1) Data Imbalance:
    - The training dataset (unredactor.tsv) may have an uneven distribution of names and contexts, which can cause the model to struggle with underrepresented classes. For example, names that appear only once or in limited contexts may lead to reduced model accuracy.
Impact: This contributes to low metric scores such as precision and recall, as the model may misclassify or fail to predict some names correctly.
2) Complexity of Redacted Contexts:
    - The context in the test.tsv file may include highly varied language structures, making it difficult for the model to generalize from the training data. Furthermore, the names may not appear in the same linguistic structure or context as in the training dataset.
Impact: This mismatch reduces the model's ability to accurately reconstruct names, especially in interdependent and complex textual scenarios.
3) TF-IDF and Random Forest Model Limitations:
    - While TF-IDF is excellent for extracting features and Random Forest is a robust classifier, these methods may not fully capture deep semantic relationships or interdependencies in textual data. More advanced deep learning models like transformers (e.g., BERT) may perform better for such tasks but require significantly more computational resources.
    - The combination of TF-IDF and Random Forest can miss nuanced relationships, particularly in challenging contexts with limited explicit patterns.
4) Limited Contextual Information in Test Data:
    - The test.tsv file often contains contexts with minimal clues about the redacted names. For example, if the name is heavily dependent on external knowledge or subtle intertextual clues, the model is unable to predict it accurately.
    - This leads to gaps in the submission file, with incorrect or blank predictions.
5) Overfitting to Training Data:
    - The model may perform well on the training and validation data (unredactor.tsv) but generalize poorly to the unseen test.tsv data due to overfitting to specific patterns in the training set.
    - While the metric scores may indicate reasonable performance on validation data, they do not translate directly to real-world performance on the test data.
6) Interpretation of Metric Scores:
    - Low metric scores reflect the inherent difficulty of the task, particularly with disjointed training and testing datasets. The goal is to iteratively refine the model and address these limitations.
7) Utility of Partial Predictions:
    - Even though the model may not achieve perfect accuracy, partial predictions can still provide meaningful insights, especially in scenarios where reconstructing any part of the redacted data is valuable.

## Bugs and Assumptions
1) Handling Empty or Incorrect Contexts: The function refine_test_predictions assumes every context will contain valid information. If the context is empty or poorly formatted, it might result in incorrect predictions or crashes.
2) Single Redaction Per Context: Assumes that each context contains only one redacted name to predict. Multiple redactions in a single context might not be handled.
3) Assumption: Assumes that TF-IDF vectorization is enough to capture the nuances in the text for predicting names. However, semantic meaning (e.g., relationships between words) might not be fully captured.
4) Ambiguity in Context: Many contexts in the dataset are highly ambiguous and do not provide sufficient information to accurately predict the redacted name. For example, generic sentences without specific references can confuse the model.
5) Assumption: The surrounding text in the context column provides enough information to uniquely identify the redacted name. This can create ambiguity or generic contexts can reduce prediction accuracy.