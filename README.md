# Embedding Playground
## Me teaching me about embeddings and  the gensim lib

## Install
All examples are built off the 50 dimension, 6B glove embeddings.  To prepare your working directory:
- Download <https://nlp.stanford.edu/data/glove.6B.zip> from <https://nlp.stanford.edu/projects/glove/>
- unzip the file.
- scripts assume the presence of ```glove.6B.50d.txt``` in your directory.  All other versions can be deleted.

Other than that, the normal process applies:
- Create your venv (I'm using python 3.12 here)
- Run ```pip install -r requirements.txt```

## About the files

### graph_embeddings.py
Take a list of words, looks up their embeddings and then uses TSNE for dimensionality reduction to 2 engineered dimensions.  These are then used to graph the words.

THis is useful as it shows how the embedding space is common to all words, and that similar ideas tend to cluster there.

### vector_math.py
Attempts to provethe addage that in the embeggings space ('king' - 'man') + 'woman' = 'queen' (hint: It doesn't).  But still, looking at how these vecotrs work and can be related to one another is interesting.

### word_distances.py
Takes a list of terms and returns a heat map of the euclidian distances between them all.  This differs from the graph in that no dimensenionality reduction is performed, only a simple euclidian distance calculation.

### gensim_tutorials
Taken from <https://radimrehurek.com/gensim/auto_examples/>.  For learning about the gensim library