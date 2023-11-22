# https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html#sphx-glr-auto-examples-core-run-core-concepts-py

import pprint
from gensim import corpora
from gensim import models
from gensim import similarities

# a document is a single str
document = "Human machine interface for lab abc computer applications"

# a corpus is a collection of documents
text_corpus = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement",
    "The generation of random binary unordered trees",
    "The intersection graph of paths in trees",
    "Graph minors IV Widths of trees and well quasi ordering",
    "Graph minors A survey",
]

# primative preprocessing
# see gensim.utils.simple_preprocess for better

# Create a set of frequent words
stoplist = set('for a of the and to in'.split(' '))
# Lowercase each document, split it by white space and filter out stopwords
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in text_corpus]

# Count word frequencies
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Only keep words that appear more than once
processed_corpus = [[token for token in text if frequency[token] > 1] for text in texts]
# pprint.pprint(processed_corpus)

# Corpora module.  Takes a list of documents and associates
# each word with an int vale
dictionary = corpora.Dictionary(processed_corpus)
# print(dictionary)

# Vectorization can be use-case specific.
# In this case each feature is represented as a dense vector,
# i.e. a series of question/answer pairs
# for example:
# Q1: How many times does the word splonge appear in the doc
# A1: 0
# Q2: How many paragraphs does the document contain?
# A2: 2
# Q3: How many fonts does the documents contain?
# A3: 5
# would be vectorized as:
# (1,0.0), (2,2.0), (3,5.0)
# To save memory, gensim will eliminate entries where the value = 0
# Meaning that this would be encoded as (2,2.0), (3,5.0)
# The other option is to encode as a 'bag of words'
# Wherein dict ['coffee','milk','sugar','spoon']
# would be used to encode the string "coffee milk coffee"
# as [2, 1, 0, 0]

# since the dorpus above contains 12 words, 
# it is a 12-dimensional vector
# pprint.pprint(dictionary.token2id)

# we can use doc2bow to encode new documents using this dict
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
#print(new_vec)
# returns [(0, 1), (1, 1)] meaning that: 
# the dictionaty token 0 (computer) appeard once in the doc
# and dictionaty token 1 (human) also appears once in the doc

#Based on this, our prior corpus becomes:
bow_corpus = [dictionary.doc2bow(text) for text in processed_corpus]
# pprint.pprint(bow_corpus)

# Once the corpus is vectorized, it can be modeled
# tf-idf transforms from bow to a vector space where 
# the freq counts are weighted according to the relative 
# rairty of each word

# train the model
tfidf = models.TfidfModel(bow_corpus)

# transform "system minors" string
words = "system minors".lower().split()
#print(tfidf[dictionary.doc2bow(words)])
# returns [(5, 0.5898341626740045), (11, 0.8075244024440723)]
# meaning that system 
# (dict index 5, 4 appearances in corpus) 
# has less weight than minors 
# (dict index 11, 2 appearances)

#once the corpus is modeled, you can index it for similarities
index = similarities.SparseMatrixSimilarity(tfidf[bow_corpus], 
                                            num_features=12)

# and query a document agianst the entire corpus
query_document = 'system engineering'.split()
query_bow = dictionary.doc2bow(query_document)
sims = index[tfidf[query_bow]]
#print(list(enumerate(sims)))
# [(0, 0.0), (1, 0.32448703), (2, 0.41707572), (3, 0.7184812), (4, 0.0), (5, 0.0), (6, 0.0), (7, 0.0), (8, 0.0)]
# means that the string is 32% similar to document id 1
# (document number 2)
for document_number, score in sorted(enumerate(sims), key=lambda x: x[1], reverse=True):
    print(document_number, score)