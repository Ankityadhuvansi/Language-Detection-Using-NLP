
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("Language Detection.csv")

value_count = data["Language"].value_counts()

independent = data["Text"]
dependent = data["Language"]

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
dependent = label_encoder.fit_transform(dependent)

data_list = []


for text in independent:
    # remove all symbols and numbers (if any)
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r"[[]]", ' ', text)

    text = text.lower()

    data_list.append(text)

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
independent = cv.fit_transform(data_list).toarray()
var = independent.shape

from sklearn.model_selection import train_test_split

i_train, i_test, d_train, d_test = train_test_split(independent, dependent, test_size=0.2)


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(i_train, d_train)


d_pred = model.predict(i_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ac = accuracy_score(d_test, d_pred)
cm = confusion_matrix(d_test, d_pred)
print("Accuracy is :", ac)

# confusion Matrix
plt.figure(figsize=(15, 10))
sns.heatmap(cm, annot=True)
plt.show()


def predict(text):
    independent = cv.transform([text]).toarray()  # convert text to bag of words model
    lang = model.predict(independent)  # predict the language
    lang = label_encoder.inverse_transform(lang)  # find the language that relates to the predicted value
    print("The langauge is in", lang[0])  # print the language!


predict("español")  # Enter the Text to Predict
