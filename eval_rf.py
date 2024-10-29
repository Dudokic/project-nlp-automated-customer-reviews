import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from joblib import load

# Load your dataset
df = pd.read_csv('./Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv')

# Convert ratings to sentiment labels
def convert_to_sentiment(score):
    if score in [1, 2, 3]:
        return 'negative'
    elif score == 4:
        return 'neutral'
    else:
        return 'positive'

df['sentiment'] = df['reviews.rating'].apply(convert_to_sentiment)

# Text Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in sklearn_stopwords]
    return ' '.join(words)

df['cleaned_text'] = df['reviews.text'].apply(clean_text)

# Load Random Forest model and TF-IDF vectorizer
rf_model = load('./random_forest_model_f.joblib')
vectorizer = load('./tfidf_vectorizer.joblib')  # Load the pre-fitted vectorizer

# Transform text for Random Forest model using the loaded TF-IDF vectorizer
X_rf = vectorizer.transform(df['cleaned_text'])

# Predictions for Random Forest model (these will be strings: 'negative', 'neutral', 'positive')
rf_predicted_labels = rf_model.predict(X_rf)

# True labels as strings
true_labels = df['sentiment'].tolist()  # Use string labels for consistency

# Define a function to calculate evaluation metrics
def evaluate_model(true_labels, predicted_labels, model_name):
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print(f"\n---{model_name} Evaluation---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:\n", conf_matrix)
    print("\nClassification Report:\n", classification_report(true_labels, predicted_labels, target_names=['negative', 'neutral', 'positive']))

# Evaluate the Random Forest model
evaluate_model(true_labels, rf_predicted_labels, "Random Forest Model")
