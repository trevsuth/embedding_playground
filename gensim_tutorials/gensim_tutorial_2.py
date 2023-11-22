# https://radimrehurek.com/gensim/auto_examples/core/run_corpora_and_vector_spaces.html#sphx-glr-auto-examples-core-run-corpora-and-vector-spaces-py

import logging
from pprint import pprint
from collections import defaultdict
from gensim import corpora
from smart_open import open  # for transparently opening remote files
import gensim
import numpy as np
import scipy.sparse

# set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# create a small corpus of 9 documents
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

# pre-process corpus using a toy stop list
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

#pprint(texts)

# create a corpora, save it to /tmp
dictionary = corpora.Dictionary(texts)
#dictionary.save('/tmp/deerwester.dict')  # store the dictionary, for future reference
#dictionary.save('./gensim_tutorials/deerwester.dict')  # store the dictionary, for future reference
#print(dictionary)
#print(dictionary.token2id)

# convert a new doc to bow vectors
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
#print(new_vec)  # the word "interaction" does not appear in the dictionary and is ignored

#convert the entire corpus
corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)  # store to disk, for later use
#print(corpus)

#Streaming documents
class MyCorpus:
    def __iter__(self):
        for line in open('https://radimrehurek.com/mycorpus.txt'):
            # assume there's one document per line, tokens separated by whitespace
            yield dictionary.doc2bow(line.lower().split())

#gensim can take any object that can be iterated over, be it lists, np, pd, etc.
#it does assume that in a .txt, each doc is its own line

corpus_memory_friendly = MyCorpus()  # doesn't load the corpus into memory!
#print(corpus_memory_friendly)
# or, to pretty(er) print, 

#for vector in corpus_memory_friendly:  # load one vector into memory at a time
#    print(vector)

# loading the vecor this way only loads 1 document into memory at a time

# you can also make the dictionary in a similar way
# collect statistics about all tokens
dictionary = corpora.Dictionary(line.lower().split() for line in open('https://radimrehurek.com/mycorpus.txt'))
# remove stop words and words that appear only once
stop_ids = [
    dictionary.token2id[stopword]
    for stopword in stoplist
    if stopword in dictionary.token2id
]
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
dictionary.filter_tokens(stop_ids + once_ids)  # remove stop words and words that appear only once
dictionary.compactify()  # remove gaps in id sequence after words that were removed
print(dictionary)

#Corpus formats

#Market Matrix format
corpus = [[(1, 0.5)], []]  # make one document empty, for the heck of it

corpora.MmCorpus.serialize('/tmp/corpus.mm', corpus)
# returns
# 2022-04-22 19:16:05,705 : INFO : storing corpus in Matrix Market format to /tmp/corpus.mm
# 2022-04-22 19:16:05,708 : INFO : saving sparse matrix to /tmp/corpus.mm
# 2022-04-22 19:16:05,708 : INFO : PROGRESS: saving document #0
# 2022-04-22 19:16:05,708 : INFO : saved 2x2 matrix, density=25.000% (1/4)
# 2022-04-22 19:16:05,709 : INFO : saving MmCorpus index to /tmp/corpus.mm.index

# Other formats
corpora.SvmLightCorpus.serialize('/tmp/corpus.svmlight', corpus)
corpora.BleiCorpus.serialize('/tmp/corpus.lda-c', corpus)
corpora.LowCorpus.serialize('/tmp/corpus.low', corpus)

# 2022-04-22 19:16:05,818 : INFO : converting corpus to SVMlight format: /tmp/corpus.svmlight
# 2022-04-22 19:16:05,820 : INFO : saving SvmLightCorpus index to /tmp/corpus.svmlight.index
# 2022-04-22 19:16:05,821 : INFO : no word id mapping provided; initializing from corpus
# 2022-04-22 19:16:05,821 : INFO : storing corpus in Blei's LDA-C format into /tmp/corpus.lda-c
# 2022-04-22 19:16:05,821 : INFO : saving vocabulary of 2 words to /tmp/corpus.lda-c.vocab
# 2022-04-22 19:16:05,822 : INFO : saving BleiCorpus index to /tmp/corpus.lda-c.index
# 2022-04-22 19:16:05,934 : INFO : no word id mapping provided; initializing from corpus
# 2022-04-22 19:16:05,936 : INFO : storing corpus in List-Of-Words format into /tmp/corpus.low
# 2022-04-22 19:16:05,937 : WARNING : List-of-words format can only save vectors with integer elements; 1 float entries were truncated to integer value
# 2022-04-22 19:16:05,937 : INFO : saving LowCorpus index to /tmp/corpus.low.index

#to load it FROM a market Matrix file
corpus = corpora.MmCorpus('/tmp/corpus.mm')
#print(corpus)

# one way of printing a corpus: load it entirely into memory
#print(list(corpus))  # calling list() will convert any sequence to a plain Python list

# another way of doing it: print one document at a time, making use of the streaming interface
#for doc in corpus:
#    print(doc)

# re-save the corpus Blei's LDA-C format
corpora.BleiCorpus.serialize('/tmp/corpus.lda-c', corpus)

#working with numpy and scipy
numpy_matrix = np.random.randint(10, size=[5, 2])  # random matrix as an example
corpus = gensim.matutils.Dense2Corpus(numpy_matrix)
# numpy_matrix = gensim.matutils.corpus2dense(corpus, num_terms=number_of_corpus_features)

scipy_sparse_matrix = scipy.sparse.random(5, 2)  # random sparse matrix as example
corpus = gensim.matutils.Sparse2Corpus(scipy_sparse_matrix)
scipy_csc_matrix = gensim.matutils.corpus2csc(corpus)