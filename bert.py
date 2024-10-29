from flask import Flask, render_template, request, url_for
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords

app = Flask(__name__)

# Load the trained DistilBERT model and tokenizer
bert_model_path = './Bert'
tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_path)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in sklearn_stopwords]
    return ' '.join(words)

# Custom function for BERT prediction
def predict_sentiment(text):
    # Tokenize input without `token_type_ids`
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        prediction = torch.argmax(outputs.logits, dim=1).item()
    return prediction

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the user input
        user_input = request.form['review']
        cleaned_text = clean_text(user_input)
        
        # Make prediction
        prediction = predict_sentiment(cleaned_text)
        
        # Map prediction to sentiment and set appropriate GIF
        if prediction == 2:  # Assuming label 2 is positive
            result = '😊 Positive'
            gif_path = url_for('static', filename='happy.gif')
        elif prediction == 1:  # Assuming label 1 is neutral
            result = '😐 Neutral'
            gif_path = url_for('static', filename='neutral.gif')
        else:  # Assuming label 0 is negative
            result = '☹️ Negative'
            gif_path = url_for('static', filename='unhappy.gif')

        return render_template('index.html', review=user_input, result=result, gif_path=gif_path)

if __name__ == '__main__':
    app.run(debug=True)
