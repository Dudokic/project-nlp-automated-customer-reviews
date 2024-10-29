# In app.py
from flask import Flask, render_template, request, url_for
from joblib import load
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as sklearn_stopwords

app = Flask(__name__)

# Load the trained Random Forest model and TF-IDF vectorizer
rf_model = load('random_forest_model_f.joblib')
vectorizer = load('tfidf_vectorizer.joblib')

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in sklearn_stopwords]
    return ' '.join(words)

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
        
        # Transform input text with the loaded TF-IDF vectorizer
        vectorized_input = vectorizer.transform([cleaned_text])
        
        # Predict sentiment
        prediction = rf_model.predict(vectorized_input)[0]
        
        # Set GIF path based on the result
        if prediction == 'positive':
            gif_path = url_for('static', filename='happy.gif')
        elif prediction == 'neutral':
            gif_path = url_for('static', filename='neutral.gif')
        else:
            gif_path = url_for('static', filename='unhappy.gif')
        
        # Map prediction result to a display message
        result = {
            'positive': '😊 Positive',
            'neutral': '😐 Neutral',
            'negative': '☹️ Negative'
        }.get(prediction, "Unknown")

        return render_template('index.html', review=user_input, result=result, gif_path=gif_path)

if __name__ == '__main__':
    app.run(debug=True)