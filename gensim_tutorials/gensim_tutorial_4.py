import logging
from collections import defaultdict
from gensim import corpora
from gensim import models
from gensim import similarities

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Create the corpus -  we've done this already a few times
documents = [
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

# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [
    [word for word in document.lower().split() if word not in stoplist]
    for document in documents
]

# remove words that appear only once
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [
    [token for token in text if frequency[token] > 1]
    for text in texts
]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Similarity Interface
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)

# Use the LSI to analyze a novel document
doc = "Human computer interaction"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow]  # convert the query to LSI space
#print(vec_lsi)

#Initializing query structures
# Create index of documents
index = similarities.MatrixSimilarity(lsi[corpus])  # transform corpus to LSI space and index it
# The class similarities.MatrixSimilarity is only 
# appropriate when the whole set of vectors fits in
# to memory. For example, a corpus of one million 
# documents would require 2GB of RAM in a 
# 256-dimensional LSI space, when used with this 
# class.

# Without 2GB of free RAM, you would need to use the 
# similarities.Similarity class. This class operates 
# in fixed memory, by splitting the index across 
# multiple files on disk, called shards. It uses 
# similarities.MatrixSimilarity and similarities.
# SparseMatrixSimilarity internally, so it is still 
# fast, although slightly more complex.


# save index
index.save('/tmp/deerwester.index')
index = similarities.MatrixSimilarity.load('/tmp/deerwester.index')

# obtain similarity
sims = index[vec_lsi]  # perform a similarity query against the corpus
#print(list(enumerate(sims)))  # print (document_number, document_similarity) 2-tuples

# sort by similarities
sims = sorted(enumerate(sims), key=lambda item: -item[1])
#for doc_position, doc_score in sims:
#    print(doc_score, documents[doc_position])