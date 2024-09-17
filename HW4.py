#STEP 1: Train a word2vec model

import os
from gensim.test.utils import datapath
from gensim import utils

folder_path = '/Users/chiratidzo/Downloads/TRN'

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __init__(self, folder_path):
        self.folder_path = folder_path
        
    def __iter__(self):
        for filename in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, filename)
            if os.path.isfile(file_path):
                with open(file_path, "r", encoding="latin-1") as file:
                    for line in file:
                        #assume there's one document per line, tokens separated by whitespace
                        yield utils.simple_preprocess(line)

import gensim.models
sentences = MyCorpus(folder_path)
model = gensim.models.Word2Vec(sentences=sentences)
#train the model
model.train(sentences, total_examples=model.corpus_count, epochs=10)  

#access word vectors using model.wv
vec_king = model.wv['king']  

#top 5 most similar words to 'king'
similar_words = model.wv.most_similar('king', topn=5)  
for word, similarity in similar_words:
    print(f"Similar word: {word}, Similarity score: {similarity:.4f}")


pairs = [
    ('king', 'queen'),
    ('king', 'prince'), 
    ('king', 'horse'),
    ('king', 'car'),    
    ('king', 'cake'),
    ('king', 'man'),
]
for w1, w2 in pairs:
    print('%r\t%r\t%.2f' % (w1, w2, model.wv.similarity(w1, w2)))
