from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def load_embeddings():
    # file names
    glove_input_file = 'glove.6B.50d.txt'
    
    # load model
    model = KeyedVectors.load_word2vec_format(glove_input_file, 
                                          binary=False,
                                          no_header=True)
    return model

def reduce_dimensions(word_vector, num_components=2):
    tsne = TSNE(n_components=num_components, 
                random_state=0)
    new_vector = tsne.fit_transform(word_vector)
    return new_vector

model = load_embeddings()
#vector = model['computer']
#print(vector)

#similar = model.most_similar('computer', 5)
#print(similar)

# Select a subset of word embeddings (for better visualization)
words = ['king', 'queen', 'man', 'woman', 'city', 'town', 'car', 'bike']
vectors = [model[word] for word in words]

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=0)
vectors_array = np.vstack(vectors)  # Convert list of vectors to a numpy array

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=0, perplexity=1)
vectors_2d = tsne.fit_transform(vectors_array)  # Use the numpy array here

# Plot
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])
for i, word in enumerate(words):
    plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]))
plt.show()