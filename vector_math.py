from gensim.models import KeyedVectors

def load_glove_model(glove_file_path):
    # Load the GloVe model
    model = KeyedVectors.load_word2vec_format(glove_file_path, binary=False, no_header=True)
    return model


# Load your GloVe model
model = load_glove_model('DataSets/glove.6B.50d.txt')

# Get vectors
king = model['king']
queen = model['queen']
man = model['man']
woman = model['woman']

# DO the math
calculated_vector = (king - man) + woman

# Find the most similar words
most_similar_words = model.similar_by_vector(calculated_vector, 
                                             topn=5)
print(most_similar_words)