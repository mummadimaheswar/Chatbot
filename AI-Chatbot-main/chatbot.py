# chatbot.py

import json
import random
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
nltk.download('punkt')
with open('intents.json') as file:
    data = json.load(file)
patterns = []
tags = []
responses = {}
for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])
    responses[intent['tag']] = intent['responses']
vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize)
X = vectorizer.fit_transform(patterns)
y = np.array(tags)
clf = MultinomialNB()
clf.fit(X, y)
def predict_intent(text):
    X_test = vectorizer.transform([text])
    pred_tag = clf.predict(X_test)[0]
    return pred_tag
def get_response(user_input):
    intent = predict_intent(user_input)
    return random.choice(responses[intent])
