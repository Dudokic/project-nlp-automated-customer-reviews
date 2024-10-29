# %% [markdown]
# #Data Collection and Preparation

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import classification_report


# %% [markdown]
# Load dataset

# %%
df = pd.read_csv('./Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv',
                 usecols=['reviews.text', 'reviews.rating'])


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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# %% [markdown]
# #Traditional NLP & ML Approach
# 

# %% [markdown]
# Data Preprocessing

# %%
# Vectorize the text using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# %% [markdown]
# Model Building

# %%
#Train a Naive Bayes Model:
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# %%
# Train a Logistic Regression Model:
lr_model = LogisticRegression()
lr_model.fit(X_train_tfidf, y_train)

# %%
#Train a Random Forest Model:
rf_model = RandomForestClassifier()
rf_model.fit(X_train_tfidf, y_train)


# %%
#Train an SVM Model:
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# %% [markdown]
# Model evaluation
# 

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
# #Transformer Approach (HuggingFace)

# %% [markdown]
# Data Preprocessing

# %%


#Load Pre-Trained Tokenizer and Model (e.g., BERT)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # 3 for pos, neg, neu


#Tokenize and Encode Text:

def tokenize_data(texts, tokenizer):
    return tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

X_train_tokens = tokenize_data(X_train.tolist(), tokenizer)
X_test_tokens = tokenize_data(X_test.tolist(), tokenizer)

# %% [markdown]
# Fine-tuning the Model

# %%
# Step 1: Import necessary libraries
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Step 2: Map sentiment labels to integers
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}

# Assume df['sentiment'] contains 'positive', 'neutral', 'negative'
# Optionally, sample a smaller dataset for faster experimentation
# Use the entire dataset without sampling
X = df['reviews.text']  # Text data
y = df['sentiment']     # Sentiment labels ('positive', 'neutral', 'negative')

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Map sentiment labels to integers
y_train_mapped = y_train.map(sentiment_mapping).values
y_test_mapped = y_test.map(sentiment_mapping).values

# Step 4: Load pre-trained DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the training and test sets using DistilBERT tokenizer
X_train_tokens = tokenizer(
    X_train.tolist(),  # Convert reviews into list
    padding=True,
    truncation=True,
    return_tensors='pt'  # Return PyTorch tensors
)

X_test_tokens = tokenizer(
    X_test.tolist(),
    padding=True,
    truncation=True,
    return_tensors='pt'
)

# Step 5: Convert tokenized inputs to HuggingFace Datasets
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

train_dataset = Dataset.from_dict(train_data)
test_dataset = Dataset.from_dict(test_data)

# Step 6: Load pre-trained DistilBERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

# Step 7: Define training arguments for faster training with optimizations
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    per_device_train_batch_size=32,  # Increased batch size for faster training
    per_device_eval_batch_size=32,
    num_train_epochs=10,  # Reduced number of epochs
    weight_decay=0.01,
    logging_dir='./logs',  # Directory for logging
    logging_steps=10,
    fp16=True  # Enable mixed precision training for speedup
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


# %%
#evaluate
# Evaluate the HuggingFace Transformer model
predictions = trainer.predict(test_dataset)

# Convert logits (raw predictions) to class predictions
pred_labels = np.argmax(predictions.predictions, axis=-1)

# Use y_test_mapped (integer labels) instead of y_test (string labels)
accuracy = accuracy_score(y_test_mapped, pred_labels)
print(f"Transformer Accuracy: {accuracy}")

# Detailed classification report
print("Classification Report:")
print(classification_report(y_test_mapped, pred_labels, target_names=['negative', 'neutral', 'positive']))


# %%

# Save the fine-tuned model using HuggingFace's method
model.save_pretrained('./Bert')
tokenizer.save_pretrained('./Bert')  # Save the tokenizer as well


# %% [markdown]
# #Summarization (Bonus)


