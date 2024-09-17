import pandas as pd
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


#read files and convert from list type to string
with open("rt-polarity.neg.txt","r") as file:
    negative = file.readlines()
negative = ''.join(negative)

with open("rt-polarity.pos.txt","r") as file:
    positive = file.readlines()
positive = ''.join(positive)

#remove everything that isn't a review
start_neg = negative.find('{"rawLines":[')
end_neg = negative.find('],"stylingDirectives"')
neg = negative[start_neg + len('{"rawLines":['):end_neg]
neg = neg.strip().split('","')

start_pos = positive.find('{"rawLines":[')
end_pos = positive.find('],"stylingDirectives"')
pos = positive[start_pos + + len('{"rawLines":['):end_pos]
pos = pos.strip().split('","')

#STEP 1: split into training, development and test
import random
import math

random.seed(42)

random.shuffle(neg)
neg_size = len(neg)
neg_train_size = int(0.7*neg_size)
neg_dev_size = math.ceil(0.15*neg_size)

neg_train = neg[:neg_train_size]
neg_dev = neg[neg_train_size:neg_train_size+neg_dev_size]
neg_test = neg[neg_train_size + neg_dev_size:]

random.shuffle(pos)
pos_size = len(pos)
pos_train_size = int(0.7*pos_size)
pos_dev_size = math.ceil(0.15*pos_size)

pos_train = pos[:pos_train_size]
pos_dev = pos[pos_train_size:pos_train_size+pos_dev_size]
pos_test = pos[pos_train_size + pos_dev_size:]

#STEP 2: implement vectorizer to convert training data into numpy array
#remove punctuation and make lower case
punctuation = [".", ",", ":", ";", "(", ")", "-", "?", "/", "*", "!", "#", "'", '"']

neg_train2 = []
for review in neg_train:
    review = review.lower()
    for mark in punctuation:
        review = review.replace(mark, "")
    neg_train2.append(review)

pos_train2 = []
for review in pos_train:
    review = review.lower()
    for mark in punctuation:
        review = review.replace(mark, "")
    pos_train2.append(review)
        
#get unique words
unique_words = set()
for review in neg_train2:
    words = review.split()
    unique_words.update(words)

for review in pos_train2:
    words = review.split()
    unique_words.update(words)


def review_factor(neg_reviews, pos_reviews):
    array = pd.DataFrame()
    array["Review"] = ['review{}'.format(i+1) for i in range(len(neg_reviews) + len(pos_reviews))]

    #create a list to hold the data for all words
    word_data = []

    #populate data dictionary
    for i, review in enumerate(neg_reviews + pos_reviews):
        review = review.lower()
        for mark in punctuation:
            review = review.replace(mark, "")
        words = review.split()
        word_row = {}
        # Initialize all word columns with 0
        for word in unique_words:
            word_row[word] = 0

        #initialize counter
        word_counts = Counter(words)

        for word, count in word_counts.items():
            if word in unique_words:
                word_row[word] = count

        word_row["sentiment"] = "neg" if i < len(neg_reviews) else "pos"
        word_data.append(word_row)

    # Concatenate the word data into a DataFrame
    word_df = pd.DataFrame(word_data)

    # Concatenate the review DataFrame with the word DataFrame
    array = pd.concat([word_df["sentiment"], word_df.drop("sentiment", axis=1)], axis=1)

    return array


train_array = review_factor(neg_train, pos_train)

#remove words that occur less than 5 times or more than 5000 times
column_sums = train_array.iloc[:, 1:].astype(bool).sum()
columns_to_keep = (column_sums >= 5)& (column_sums <= 5000)
columns_to_keep[train_array.columns[0]] = True

train_array = train_array.loc[:, columns_to_keep]
unique_words = list(train_array.columns[1:])
    
index_dict = {column_name: i for i, column_name in enumerate(train_array.columns)}

train_array = train_array.values

def review_vectorizer(neg_reviews, pos_reviews):
    array = pd.DataFrame()
    
    #sentiment column
    #sentiment_column = ['review{}'.format(i+1) for i in range(len(neg_reviews) + len(pos_reviews))]
    #array["sentiment"] = sentiment_column

    # Create a list to hold the data for all words
    word_data = []

    # Populate data dictionary
    for i, review in enumerate(neg_reviews + pos_reviews):
        review = review.lower()
        for mark in punctuation:
            review = review.replace(mark, "")
        words = review.split()
        word_row = {}
        #initialize all word columns with 0
        for word in unique_words:
            word_row[word] = 0

        #initialize counter
        word_counts = Counter(words)

        for word, count in word_counts.items():
            if word in unique_words:
                word_row[word] = count

        word_row["sentiment"] = "neg" if i < len(neg_reviews) else "pos"
        word_data.append(word_row)

    #concatenate the word data into a DataFrame
    word_df = pd.DataFrame(word_data)

    #concatenate the review DataFrame with the word DataFrame
    array = pd.concat([word_df["sentiment"], word_df.drop("sentiment", axis=1)], axis=1)

    return array.values

#STEP 3: classifier
classifier = LogisticRegression()

train_X = train_array[:, 1:]
#sentiment
train_y = train_array[:, 0]

#adjust hyperparameters
hyperparameters = {
    'C': 0.1,
    'penalty': 'l2',
    'solver': 'lbfgs'
}

best_classifier = LogisticRegression(**hyperparameters)


#fit on training data
best_classifier.fit(train_X, train_y)

#development daa
dev_array = review_vectorizer(neg_dev, pos_dev)

dev_X = dev_array[:, 1:]
#sentiment
dev_y = dev_array[:, 0]

predictions = best_classifier.predict(dev_X)

acc = accuracy_score(dev_y, predictions)
print('Manual development accuracy = = {:.2f}%'.format(acc * 100))

#STEP 4:
#test data
test_array = review_vectorizer(neg_test, pos_test)

test_X = test_array[:, 1:]
test_y = test_array[:, 0]

predictions = best_classifier.predict(test_X)

acc = accuracy_score(test_y, predictions)
print('Manual test accuracy = {:.2f}%'.format(acc * 100))

#STEP 5: CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier

text_clf = Pipeline([
     ('vect', CountVectorizer(stop_words='english')),
     ('tfidf', TfidfTransformer()),
     ('clf', LogisticRegression())])
text_clf.fit(neg_train + pos_train, ['neg'] * len(neg_train) + ['pos'] * len(pos_train))

dev_reviews = neg_dev + pos_dev

#true labels for the development set
true_labels = ['neg'] * len(neg_dev) + ['pos'] * len(pos_dev)

predicted = text_clf.predict(dev_reviews)
accuracy = accuracy_score(true_labels, predicted)

#adjust hyperparameters
hyperparameters = {
    'C': 1,
    'penalty': "l2",
    'solver': 'lbfgs'
}

new_classifier = Pipeline([
     ('vect', CountVectorizer(stop_words='english')),
     ('tfidf', TfidfTransformer()),
     ('clf', LogisticRegression(**hyperparameters))])

#fit the new classifier
new_classifier.fit(neg_train + pos_train, ['neg'] * len(neg_train) + ['pos'] * len(pos_train))

new_predict = new_classifier.predict(dev_reviews)
accuracy2 = accuracy_score(true_labels, new_predict)
print('Tuned CountVec development accuracy = = {:.2f}%'.format(accuracy2 * 100))

#test data
true_labels = ['neg'] * len(neg_test) + ['pos'] * len(pos_test)

test_reviews = neg_test + pos_test
test_predict = new_classifier.predict(test_reviews)
test_accuracy = accuracy_score(true_labels, test_predict)
print('Tuned CountVec test accuracy = = {:.2f}%'.format(test_accuracy * 100))

