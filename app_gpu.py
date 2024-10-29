# %% [markdown]
# # Data Collection and Preparation

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# %% [markdown]
# Load dataset

# %%
df = pd.read_csv('./Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv',
                 usecols=['reviews.text', 'reviews.rating'],
                 nrows=10000)


# %% [markdown]
# Convert Ratings to Sentiment Labels

# %%
# Select relevant columns
df = df[['reviews.text', 'reviews.rating']].copy()  # Use .copy() to avoid modifying the original DataFrame

# Drop rows with missing values in either column
df.dropna(subset=['reviews.text', 'reviews.rating'], inplace=True)

# Convert ratings to sentiment labels
def convert_to_sentiment(score):
    if score in [1, 2, 3]:
        return 'negative'
    elif score == 4:
        return 'neutral'
    else:
        return 'positive'

# Apply sentiment conversion using .loc[] to avoid SettingWithCopyWarning
df.loc[:, 'sentiment'] = df['reviews.rating'].apply(convert_to_sentiment)

# %% [markdown]
# Split Data into training and testing

# %%
# Split data into features (X) and labels (y)
X = df['reviews.text']
y = df['sentiment']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %% [markdown]
# # Traditional NLP & ML Approach

# %% [markdown]
# Data Preprocessing

# %%
# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# %% [markdown]
# Model Building

# %%
# Train a Naive Bayes Model:
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Train a Logistic Regression Model:
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)

# Train a Random Forest Model:
rf_model = RandomForestClassifier()
rf_model.fit(X_train_tfidf, y_train)

# Train an SVM Model:
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# %% [markdown]
# Model evaluation

# %%
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print(f"Confusion Matrix:\n{cm}")

evaluate_model(nb_model, X_test_tfidf, y_test)
evaluate_model(lr_model, X_test_tfidf, y_test)
evaluate_model(rf_model, X_test_tfidf, y_test)
evaluate_model(svm_model, X_test_tfidf, y_test)

# %% [markdown]
# # Transformer Approach (Hugging Face)

# %% [markdown]
# Data Preprocessing

# %%
# Check if GPU is available and set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load Pre-Trained Tokenizer and Model (e.g., BERT)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Move the model to GPU
model.to(device)

# Tokenize and Encode Text:

def tokenize_data(texts, tokenizer):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)  # Move tokens to GPU

X_train_tokens = tokenize_data(X_train.tolist(), tokenizer)
X_test_tokens = tokenize_data(X_test.tolist(), tokenizer)

# %% [markdown]
# Fine-tuning the Model

# %%
# Step 2: Map sentiment labels to integers
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}

# Map sentiment labels to integers
y_train_mapped = y_train.map(sentiment_mapping).values
y_test_mapped = y_test.map(sentiment_mapping).values

# Step 5: Convert tokenized inputs to Hugging Face Datasets
train_data = {
    'input_ids': X_train_tokens['input_ids'].tolist(),
    'attention_mask': X_train_tokens['attention_mask'].tolist(),
    'labels': list(map(int, y_train_mapped))  # Ensure labels are integers
}

test_data = {
    'input_ids': X_test_tokens['input_ids'].tolist(),
    'attention_mask': X_test_tokens['attention_mask'].tolist(),
    'labels': list(map(int, y_test_mapped))  # Ensure labels are integers
}

from datasets import Dataset

train_dataset = Dataset.from_dict(train_data)
test_dataset = Dataset.from_dict(test_data)

# Step 6: Load pre-trained DistilBERT model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3).to(device)

# Step 7: Define training arguments for faster training with optimizations
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    fp16=True  # Enable mixed precision for speedup
)

# Step 8: Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Step 9: Train the model
trainer.train()

# %% [markdown]
# Evaluation

# %%
# Evaluate the Hugging Face Transformer model
predictions = trainer.predict(test_dataset)

# Convert logits (raw predictions) to class predictions
pred_labels = np.argmax(predictions.predictions, axis=-1)

# Use y_test_mapped (integer labels) instead of y_test (string labels)
accuracy = accuracy_score(y_test_mapped, pred_labels)
print(f"Transformer Accuracy: {accuracy}")

# Detailed classification report
print("Classification Report:")
print(classification_report(y_test_mapped, pred_labels, target_names=['negative', 'neutral', 'positive']))

# %% [markdown]
# Save the fine-tuned model

# %%
# Save the fine-tuned model using HuggingFace's method
model.save_pretrained('./Bert')
tokenizer.save_pretrained('./Bert')  # Save the tokenizer as well
