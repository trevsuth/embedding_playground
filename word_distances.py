from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

def load_glove_model(glove_file_path):
    # Load the GloVe model
    model = KeyedVectors.load_word2vec_format(glove_file_path, binary=False, no_header=True)
    return model

def calculate_distances(model, words):
    distances = {}
    # Calculate distances between each pair of words
    for word1 in words:
        distances[word1] = {}
        for word2 in words:
            # Euclidean distance
            distance = euclidean(model[word1], model[word2])
            distances[word1][word2] = distance
    return distances

# Load your GloVe model
model = load_glove_model('glove.6B.50d.txt')

# Words to compare
words = ['bike', 'plane', 'car', 'bus', 'train', 'automobile', 'motorcycle']

# Calculate distances
distances = calculate_distances(model, words)

# Convert to DataFrame for pretty display
distance_df = pd.DataFrame(distances)
distance_matrix = distance_df.values

# Create a heatmap
plt.imshow(distance_matrix, 
           cmap='RdYlGn', 
           interpolation='nearest')

# Add color bar
plt.colorbar()

# Set the ticks and labels
plt.xticks(ticks=np.arange(len(words)), labels=words)
plt.yticks(ticks=np.arange(len(words)), labels=words)

# Add title and labels
plt.title("Word Vector Distances")
plt.xlabel("Words")
plt.ylabel("Words")

# Iterate over the data and create text annotations
for i in range(len(words)):
    for j in range(len(words)):
        text = plt.text(j, i, round(distance_matrix[i, j], 2),
                       ha="center", va="center", color="black")

# Show the plot
plt.show()
