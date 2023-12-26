import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def preprocess_text(text):
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r"[[]]", ' ', text)
    return text.lower()


def train_and_evaluate_model(independent, dependent):
    i_train, i_test, d_train, d_test = train_test_split(independent, dependent, test_size=0.2)

    model = MultinomialNB()
    model.fit(i_train, d_train)

    d_pred = model.predict(i_test)

    ac = accuracy_score(d_test, d_pred)
    cm = confusion_matrix(d_test, d_pred)

    print("Accuracy is:", ac)
    plt.figure(figsize=(15, 10))
    sns.heatmap(cm, annot=True)
    plt.show()

    return model


def predict_language(model, cv, text):
    independent = cv.transform([text]).toarray()
    lang = model.predict(independent)
    lang = label_encoder.inverse_transform(lang)
    print("The language is in", lang[0])


# Load dataset
data = pd.read_csv("Language Detection.csv")
independent = data["Text"]
dependent = data["Language"]

# Convert categorical variables to numerical
label_encoder = LabelEncoder()
dependent = label_encoder.fit_transform(dependent)

# Preprocess text
data_list = [preprocess_text(text) for text in independent]

# Bag of words model
cv = CountVectorizer()
independent = cv.fit_transform(data_list).toarray()

# Train and evaluate the model
trained_model = train_and_evaluate_model(independent, dependent)

# Predict language
predict_language(trained_model, cv, "español")
