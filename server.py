from flask import Flask, request, jsonify
import json
import random
import pickle
import numpy as np
from keras import models
from nltk.stem import WordNetLemmatizer
import nltk
from waitress import serve
import ssl

# Ensure NLTK downloads work without SSL errors
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('wordnet')

# Initialize chatbot
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = models.load_model('chatbot.h5')

# Initialize Flask app
app = Flask(__name__)

# CORS headers
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

# Text preprocessing
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

# Bag of Words vectorization
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Intent prediction
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": round(float(r[1]), 3)} for r in results]

# Select response
def get_response(intents_list, intents_json):
    if not intents_list:
        return "Sorry, I didn't understand that. Can you rephrase?"
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tags'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I didn't understand that. Can you rephrase?"

# Chat endpoint
@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        return "Chatbot endpoint. Please send a POST request with JSON."
    try:
        data = request.get_json(force=True)
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"response": "Please send a non-empty message."})
        intents_list = predict_class(message)
        response = get_response(intents_list, intents)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"Error processing request: {str(e)}"}), 400

# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# Run server
if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000)
