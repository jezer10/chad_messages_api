from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pandas as pd
import string

punctuation = set(string.punctuation)
spam_dataset = pd.read_csv("./spam.csv", encoding="latin-1")[["v1", "v2"]]
spam_dataset.columns = ["label", "text"]


def tokenize(sentence):
    tokens = []
    for token in sentence.split():
        new_token = []
        for character in token:
            if character not in punctuation:
                new_token.append(character.lower())
        if new_token:
            tokens.append("".join(new_token))
    return tokens


vectorizer = CountVectorizer(
    tokenizer=tokenize,
    binary=True
)

train_text, test_text, train_labels, test_labels = train_test_split(spam_dataset["text"], spam_dataset["label"],
                                                                    stratify=spam_dataset["label"])
train_X = vectorizer.fit_transform(train_text)
classifier = LinearSVC()
classifier.fit(train_X, train_labels)


def predict_messages(data):
    transformed_data = vectorizer.transform(data)
    predicts = classifier.predict(transformed_data)
    return predicts
