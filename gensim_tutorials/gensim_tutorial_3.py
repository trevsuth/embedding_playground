import logging
from collections import defaultdict
from gensim import corpora
from gensim import models
import os
import tempfile

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Create toy corpus
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

#create a transformation
tfidf = models.TfidfModel(corpus)  # step 1 -- initialize a model

# use the model to transform a single documet
doc_bow = [(0, 1), (1, 1)]
#print(tfidf[doc_bow])  # step 2 -- use the model to transform vectors

# or an entire corpus
corpus_tfidf = tfidf[corpus]
#for doc in corpus_tfidf:
    #print(doc)

# transformations can also be called on on top of another
# this transforms the tfidf model into a latent 2d space (num topics=2)
lsi_model = models.LsiModel(corpus_tfidf, 
                            id2word=dictionary, 
                            num_topics=2)  # initialize an LSI transformation
corpus_lsi = lsi_model[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

# What do these latent dimensions represent?
#lsi_model.print_topics(2)
#return:
# 2022-07-18 19:59:39,298 : INFO : topic #0(1.594): 0.703*"trees" + 0.538*"graph" + 0.402*"minors" + 0.187*"survey" + 0.061*"system" + 0.060*"time" + 0.060*"response" + 0.058*"user" + 0.049*"computer" + 0.035*"interface"
# 2022-07-18 19:59:39,298 : INFO : topic #1(1.476): -0.460*"system" + -0.373*"user" + -0.332*"eps" + -0.328*"interface" + -0.320*"time" + -0.320*"response" + -0.293*"computer" + -0.280*"human" + -0.171*"survey" + 0.161*"trees"
#
# [(0, '0.703*"trees" + 0.538*"graph" + 0.402*"minors" + 0.187*"survey" + 0.061*"system" + 0.060*"time" + 0.060*"response" + 0.058*"user" + 0.049*"computer" + 0.035*"interface"'), (1, '-0.460*"system" + -0.373*"user" + -0.332*"eps" + -0.328*"interface" + -0.320*"time" + -0.320*"response" + -0.293*"computer" + -0.280*"human" + -0.171*"survey" + 0.161*"trees"')]
# according to this, "trees", "graphs", and "minors" are 
# all related and contribute to the direction of the first
# document
# both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
#for doc, as_text in zip(corpus_lsi, documents):
#
# models can be saved and loaded with save() and load() finctions
with tempfile.NamedTemporaryFile(prefix='model-', suffix='.lsi', delete=False) as tmp:
    lsi_model.save(tmp.name)  # same for tfidf, lda, ...

loaded_lsi_model = models.LsiModel.load(tmp.name)

os.unlink(tmp.name)

# Available models

## Term Frequency * Inverse Document Frequency (Tf-Idf)
# expects bow for training and transformation
# takes a vector, returns another vector of same dimensionality
# rare features will have their value increased
# can also normalize 
model = models.TfidfModel(corpus, normalize=True)

## Okapi Best Matching, Okapi BM25 
# expects bow
# takes a vector and retuens a vecotr of the same length
# rare features have value increased
# standard ranking function used by search engines to estimate the relevance of documents to a given search query.
model = models.OkapiBM25Model(corpus)

##Latent Semantic Indexing, LSI (or sometimes LSA) 
# expects wither bow or output from tfidf model (preferree)
# transforms into a target dimensionality
model = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=300)
# also can add documents "on the fly"
model.add_documents(another_tfidf_corpus)  # now LSI has been trained on tfidf_corpus + another_tfidf_corpus
lsi_vec = model[tfidf_vec]  # convert some new document into the LSI space, without affecting the model

model.add_documents(more_documents)  # tfidf_corpus + another_tfidf_corpus + more_documents
lsi_vec = model[tfidf_vec]
#can also gradually "forget" older documents

##Random Projections RP
# Reduces vector dimensionality
# Very efficient for approximating Tlidf distances
# reccomended 100s/1000s dimensionalities
model = models.RpModel(tfidf_corpus, num_topics=500)

##Latent Dirichlet Allocation, LDA
# Expects BOW
# Non-deterministic extension of LSA
# Can be intrepterted as probability distributions over words
model = models.LdaModel(corpus, id2word=dictionary, num_topics=100)

##Hierarchical Dirichlet Process, HDP 
# non parametric (bayesian) method
# still new; use with care??
model = models.HdpModel(corpus, id2word=dictionary)