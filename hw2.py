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

#STEP 2: Train Naive Bayes Classifier

from collections import Counter
def remove_words(word_list, string_list):
    return [' '.join(filter(lambda word: word not in word_list, string.split())) for string in string_list]

words_to_remove = ['the', 'and', 'is', "of", ".", "when", "I", "you", "it", "an", "and", ",", "a", "to",
                   "is", "was", "with", "its", "it's", "be", "in", "this", "for", "has", "that", "on",
                   "are", "as", "by", "at", "from", "from", "movie", "film", "but"]

neg_train = remove_words(words_to_remove, neg_train)
pos_train = remove_words(words_to_remove, pos_train)

#define function to find conditional probabilities of words in training data set
def naive_bayes_train(train_data):
    word_counts = Counter()
    total_words = 0

    for review in train_data:
        words = review.split()
        word_counts.update(words)
        total_words = sum(word_counts.values())
        
    dictionary_train = dict(word_counts)
    #probability of each word with Laplace Smoothing
    prob = {key: (count+1)/(total_words+len(dictionary_train)) for key, count in dictionary_train.items()}
    return prob
    
#conditional probabilities of training data
neg_prob = naive_bayes_train(neg_train)
pos_prob = naive_bayes_train(pos_train)

total_neg_words = sum(len(review.split()) for review in neg_train)
total_pos_words = sum(len(review.split()) for review in pos_train)

#calculate probability of each class
prob_neg = len(neg_train)/(len(neg_train)+len(pos_train))
prob_pos = len(pos_train)/(len(neg_train)+len(pos_train))

#tune on development data
from functools import reduce

def classify(review_data):
    classifications = []
    for review in review_data:
        words = review.split()
        #probability of each word with Laplace Smoothing
        neg_conditional = [neg_prob.get(word,  (1/(len(neg_prob)+ total_neg_words))) for word in words]
        neg_probability = reduce(lambda x, y: x * y, neg_conditional)*prob_neg

        #probability of each word with Laplace Smoothing
        pos_conditional = [pos_prob.get(word,  (1/(len(pos_prob)+ total_pos_words))) for word in words]
        pos_probability = reduce(lambda x, y: x * y, pos_conditional)*prob_pos
        if neg_probability > pos_probability:
            classifications.append("negative")
        else:
            classifications.append("positive")
    return classifications

neg_class = classify(neg_dev)
neg_positives = neg_class.count("positive")
neg_negatives = neg_class.count("negative")

pos_class = classify(pos_dev)
pos_positives = pos_class.count("positive")
pos_negatives = pos_class.count("negative")

neg_accuracy = neg_negatives/(neg_negatives + neg_positives)*100
pos_accuracy = pos_positives/(pos_negatives + pos_positives)*100

accuracy = 100*(neg_negatives + pos_positives)/(neg_negatives + pos_positives + neg_positives + pos_negatives)
print("Development Accuracy: ", "{:.2f}%".format(accuracy))

#STEP 3
#train best model of training and development
neg_dev = remove_words(words_to_remove, neg_dev)
pos_dev = remove_words(words_to_remove, pos_dev)

neg_final = neg_train + neg_dev
pos_final = pos_train + pos_dev

def naive_bayes_train_best(train_data):
    word_counts = Counter()
    total_words = 0

    for review in train_data:
        words = review.split()
        word_counts.update(words)
        total_words = sum(word_counts.values())

    dictionary_train = dict(word_counts)
    #probability of each word with Laplace Smoothing
    prob = {key: (count+1)/(total_words+len(dictionary_train)) for key, count in dictionary_train.items()}
    return prob
    
#conditional probabilities of training data
neg_prob_best = naive_bayes_train_best(neg_final)
pos_prob_best = naive_bayes_train_best(pos_final)

total_neg_words_best = sum(len(review.split()) for review in neg_final)
total_pos_words_best = sum(len(review.split()) for review in pos_final)

#calculate probability of each class
prob_neg_best = len(neg_final)/(len(neg_final)+len(pos_final))
prob_pos_best = len(pos_final)/(len(neg_final)+len(pos_final))

#test on test data
def classify_best(review_data):
    classifications_best = []
    for review in review_data:
        words = review.split()
        #probability of each word with Laplace Smoothing
        neg_conditional = [neg_prob_best.get(word,  (1/(len(neg_prob_best)+ total_neg_words_best))) for word in words]
        neg_probability = reduce(lambda x, y: x * y, neg_conditional)*prob_neg_best

        #probability of each word with Laplace Smoothing
        pos_conditional = [pos_prob_best.get(word,  (1/(len(pos_prob_best)+ total_pos_words_best))) for word in words]
        pos_probability = reduce(lambda x, y: x * y, pos_conditional)*prob_pos_best
        if neg_probability > pos_probability:
            classifications_best.append("negative")
        else:
            classifications_best.append("positive")
            
    return classifications_best

neg_class = classify_best(neg_test)
neg_positives = neg_class.count("positive")
neg_negatives = neg_class.count("negative")

pos_class = classify_best(pos_test)
pos_positives = pos_class.count("positive")
pos_negatives = pos_class.count("negative")

neg_accuracy = neg_negatives/(neg_negatives + neg_positives)*100
pos_accuracy = pos_positives/(pos_negatives + pos_positives)*100

accuracy = 100*(neg_negatives + pos_positives)/(neg_negatives + pos_positives + neg_positives + pos_negatives)
print("Test Accuracy: ", "{:.2f}%".format(accuracy))

#STEP 4
#analyze classifier confidence
def class_conf(review_data):
    class_confidence = []
    for review in review_data:
        words = review.split()
        #probability of each word with Laplace Smoothing
        neg_conditional = [neg_prob_best.get(word,  (1/(len(neg_prob_best)+ total_neg_words_best))) for word in words]
        neg_probability = reduce(lambda x, y: x * y, neg_conditional)*prob_neg_best

        #probability of each word with Laplace Smoothing
        pos_conditional = [pos_prob_best.get(word,  (1/(len(pos_prob_best)+ total_pos_words_best))) for word in words]
        pos_probability = reduce(lambda x, y: x * y, pos_conditional)*prob_pos_best

        confidence = abs(pos_probability - neg_probability)
        class_confidence.append((review))

        #sort in descending order
        class_confidence.sort(key=lambda x: x[1], reverse=True)
    return class_confidence

most_conf_neg = class_conf(neg_test)[:5]
least_conf_neg = class_conf(neg_test)[-5:]

most_conf_pos = class_conf(pos_test)[:5]
least_conf_pos = class_conf(pos_test)[-5:]

#STEP 5
def useful_features(prob_dict):
    #sort the conditional probabilities in descending order
    sorted_probabilities = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    #return the top 10 features
    return sorted_probabilities[:20]

useful_neg_features = useful_features(neg_prob_best)
useful_pos_features = useful_features(pos_prob_best)

