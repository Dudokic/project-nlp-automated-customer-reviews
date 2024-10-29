import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from joblib import load
import torch
from datasets import Dataset

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
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['sentiment'].map(sentiment_mapping)

# Text Cleaning function (without NLTK)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in sklearn_stopwords]
    return ' '.join(words)

df['cleaned_text'] = df['reviews.text'].apply(clean_text)


# # BERT Model Evaluation Setup

# Load DistilBERT tokenizer and model
bert_model_path = './Bert'
tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_path)

# Tokenize the data
X_test_tokens = tokenizer(
    df['reviews.text'].tolist(),
    padding=True,
    truncation=True,
    return_tensors='pt'
)

# Prepare Hugging Face Dataset for Trainer
test_data = {
    'input_ids': X_test_tokens['input_ids'],
    'attention_mask': X_test_tokens['attention_mask'],
    'labels': torch.tensor(df['label'].values)
}
test_dataset = Dataset.from_dict(test_data)

# Set up Trainer for predictions
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
training_args = TrainingArguments(output_dir='./results', per_device_eval_batch_size=32)
trainer = Trainer(model=bert_model.to(device), args=training_args)

# Predict using DistilBERT
bert_outputs = trainer.predict(test_dataset)
bert_predicted_labels = torch.argmax(torch.tensor(bert_outputs.predictions), axis=1).numpy()

# True labels
true_labels = df['label'].tolist()

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

# Evaluate both models
evaluate_model(true_labels, bert_predicted_labels, "DistilBERT Model")

