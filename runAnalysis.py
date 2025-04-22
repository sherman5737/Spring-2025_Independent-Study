
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS 
from collections import Counter
import re

models = ['GPT2', 'BERT', 'MPNET', 'MINI', 'T5', 'word2vec', 'BGE', 'ROBERTA', 'ALBERT', 'XLNet']

# Spearman correlation matrix
spearman_matrix_dice = np.array([
    [0.0,   0.433, 0.124, 0.105, 0.521, 0.432, 0.31,  0.205, 0.532, 0.238],
    [0.433, 0.0,   0.258, 0.181, 0.849, 0.815, 0.495, 0.467, 0.832, 0.439],
    [0.124, 0.258, 0.0,   0.719, 0.184, 0.16,  0.578, 0.67,  0.253, 0.128],
    [0.105, 0.181, 0.719, 0.0,   0.111, 0.077, 0.539, 0.545, 0.191, 0.052],
    [0.521, 0.849, 0.184, 0.111, 0.0,   0.868, 0.444, 0.388, 0.911, 0.463],
    [0.432, 0.815, 0.16,  0.077, 0.868, 0.0,   0.399, 0.36,  0.829, 0.444],
    [0.31,  0.495, 0.578, 0.539, 0.444, 0.399, 0.0,   0.616, 0.496, 0.225],
    [0.205, 0.467, 0.67,  0.545, 0.388, 0.36,  0.616, 0.0,   0.421, 0.285],
    [0.532, 0.832, 0.253, 0.191, 0.911, 0.829, 0.496, 0.421, 0.0,   0.435],
    [0.238, 0.439, 0.128, 0.052, 0.463, 0.444, 0.225, 0.285, 0.435, 0.0]
])

spearman_matrix_gen = np.array([
    [0.0,   0.424, 0.397, 0.342, 0.458, 0.376, 0.35,  0.382, 0.419, 0.267],
    [0.424, 0.0,   0.742, 0.68,  0.87,  0.741, 0.584, 0.731, 0.836, 0.513],
    [0.397, 0.742, 0.0,   0.882, 0.738, 0.617, 0.729, 0.915, 0.706, 0.428],
    [0.342, 0.68,  0.882, 0.0,   0.68,  0.584, 0.745, 0.854, 0.654, 0.379],
    [0.458, 0.87,  0.738, 0.68,  0.0,   0.799, 0.592, 0.722, 0.895, 0.532],
    [0.376, 0.741, 0.617, 0.584, 0.799, 0.0,   0.494, 0.599, 0.756, 0.465],
    [0.35,  0.584, 0.729, 0.745, 0.592, 0.494, 0.0,   0.707, 0.577, 0.338],
    [0.382, 0.731, 0.915, 0.854, 0.722, 0.599, 0.707, 0.0,   0.7,   0.417],
    [0.419, 0.836, 0.706, 0.654, 0.895, 0.756, 0.577, 0.7,   0.0,   0.511],
    [0.267, 0.513, 0.428, 0.379, 0.532, 0.465, 0.338, 0.417, 0.511, 0.0]
])

spearman_matrix_gest = np.array([
    [0.0,   0.106, 0.071, 0.061, 0.171, 0.149, 0.09,  0.062, 0.169, 0.075],
    [0.106, 0.0,   0.288, 0.292, 0.492, 0.31,  0.22,  0.29,  0.563, 0.153],
    [0.071, 0.288, 0.0,   0.695, 0.288, 0.038, 0.607, 0.794, 0.26,  0.083],
    [0.061, 0.292, 0.695, 0.0,   0.323, 0.081, 0.521, 0.651, 0.3,   0.091],
    [0.171, 0.492, 0.288, 0.323, 0.0,   0.506, 0.21,  0.28,  0.636, 0.171],
    [0.149, 0.31,  0.038, 0.081, 0.506, 0.0,  -0.066, 0.046, 0.481, 0.139],
    [0.09,  0.22,  0.607, 0.521, 0.21, -0.066, 0.0,   0.581, 0.205, 0.101],
    [0.062, 0.29,  0.794, 0.651, 0.28,  0.046, 0.581, 0.0,   0.255, 0.073],
    [0.169, 0.563, 0.26,  0.3,   0.636, 0.481, 0.205, 0.255, 0.0,   0.217],
    [0.075, 0.153, 0.083, 0.091, 0.171, 0.139, 0.101, 0.073, 0.217, 0.0]
])

spearman_matrix_jbb = np.array([
    [0.0, 0.22, 0.095, 0.115, 0.252, 0.18, 0.167, 0.064, 0.266, 0.05],
    [0.22, 0.0, 0.506, 0.478, 0.527, 0.381, 0.408, 0.513, 0.633, 0.092],
    [0.095, 0.506, 0.0, 0.754, 0.482, 0.302, 0.601, 0.812, 0.446, 0.122],
    [0.115, 0.478, 0.754, 0.0, 0.466, 0.265, 0.605, 0.659, 0.453, 0.142],
    [0.252, 0.527, 0.482, 0.466, 0.0, 0.424, 0.45, 0.484, 0.65, 0.13],
    [0.18, 0.381, 0.302, 0.265, 0.424, 0.0, 0.215, 0.335, 0.424, 0.031],
    [0.167, 0.408, 0.601, 0.605, 0.45, 0.215, 0.0, 0.512, 0.449, 0.098],
    [0.064, 0.513, 0.812, 0.659, 0.484, 0.335, 0.512, 0.0, 0.427, 0.092],
    [0.266, 0.633, 0.446, 0.453, 0.65, 0.424, 0.449, 0.427, 0.0, 0.125],
    [0.05, 0.092, 0.122, 0.142, 0.13, 0.031, 0.098, 0.092, 0.125, 0.0]
])

spearman_matrix_mteb = np.array([
    [0.0, 0.233, 0.135, 0.13, 0.247, 0.114, 0.031, 0.156, 0.307, 0.103],
    [0.233, 0.0, 0.255, 0.26, 0.595, 0.518, 0.259, 0.32, 0.79, 0.086],
    [0.135, 0.255, 0.0, 0.649, 0.24, 0.084, 0.329, 0.686, 0.257, 0.03],
    [0.13, 0.26, 0.649, 0.0, 0.231, 0.089, 0.327, 0.581, 0.264, 0.034],
    [0.247, 0.595, 0.24, 0.231, 0.0, 0.465, 0.173, 0.285, 0.64, 0.089],
    [0.114, 0.518, 0.084, 0.089, 0.465, 0.0, 0.198, 0.134, 0.575, -0.007],
    [0.031, 0.259, 0.329, 0.327, 0.173, 0.198, 0.0, 0.359, 0.286, 0.013],
    [0.156, 0.32, 0.686, 0.581, 0.285, 0.134, 0.359, 0.0, 0.326, 0.036],
    [0.307, 0.79, 0.257, 0.264, 0.64, 0.575, 0.286, 0.326, 0.0, 0.101],
    [0.103, 0.086, 0.03, 0.034, 0.089, -0.007, 0.013, 0.036, 0.101, 0.0]
])

spearman_matrix_prism = np.array([
    [0.0, 0.261, 0.071, 0.077, 0.303, 0.201, 0.077, 0.092, 0.347, 0.158],
    [0.261, 0.0, 0.239, 0.259, 0.665, 0.5, 0.224, 0.253, 0.703, 0.2],
    [0.071, 0.239, 0.0, 0.716, 0.136, 0.017, 0.5, 0.754, 0.223, 0.055],
    [0.077, 0.259, 0.716, 0.0, 0.158, 0.044, 0.502, 0.659, 0.244, 0.055],
    [0.303, 0.665, 0.136, 0.158, 0.0, 0.684, 0.176, 0.153, 0.714, 0.242],
    [0.201, 0.5, 0.017, 0.044, 0.684, 0.0, 0.113, 0.02, 0.561, 0.171],
    [0.077, 0.224, 0.5, 0.502, 0.176, 0.113, 0.0, 0.483, 0.275, 0.06],
    [0.092, 0.253, 0.754, 0.659, 0.153, 0.02, 0.483, 0.0, 0.234, 0.061],
    [0.347, 0.703, 0.223, 0.244, 0.714, 0.561, 0.275, 0.234, 0.0, 0.236],
    [0.158, 0.2, 0.055, 0.055, 0.242, 0.171, 0.06, 0.061, 0.236, 0.0]
]) 

spearman_matrix_safe = np.array([
    [0.0, 0.154, 0.113, 0.107, 0.212, 0.081, 0.132, 0.11, 0.258, 0.088],
    [0.154, 0.0, 0.4, 0.413, 0.517, 0.192, 0.35, 0.405, 0.591, 0.164],
    [0.113, 0.4, 0.0, 0.737, 0.372, 0.187, 0.54, 0.766, 0.39, 0.137],
    [0.107, 0.413, 0.737, 0.0, 0.36, 0.185, 0.529, 0.673, 0.401, 0.127],
    [0.212, 0.517, 0.372, 0.36, 0.0, 0.321, 0.381, 0.379, 0.656, 0.262],
    [0.081, 0.192, 0.187, 0.185, 0.321, 0.0, 0.092, 0.16, 0.319, 0.127],
    [0.132, 0.35, 0.54, 0.529, 0.381, 0.092, 0.0, 0.474, 0.396, 0.142],
    [0.11, 0.405, 0.766, 0.673, 0.379, 0.16, 0.474, 0.0, 0.374, 0.138],
    [0.258, 0.591, 0.39, 0.401, 0.656, 0.319, 0.396, 0.374, 0.0, 0.227],
    [0.088, 0.164, 0.137, 0.127, 0.262, 0.127, 0.142, 0.138, 0.227, 0.0]
])

spearman_matrix_sg = np.array([
    [0.0, 0.252, 0.217, 0.2, 0.3, 0.079, 0.231, 0.207, 0.285, 0.129],
    [0.252, 0.0, 0.539, 0.531, 0.683, 0.334, 0.554, 0.521, 0.686, 0.291],
    [0.217, 0.539, 0.0, 0.827, 0.488, 0.028, 0.826, 0.899, 0.444, 0.239],
    [0.2, 0.531, 0.827, 0.0, 0.474, 0.09, 0.776, 0.786, 0.456, 0.22],
    [0.3, 0.683, 0.488, 0.474, 0.0, 0.431, 0.536, 0.476, 0.714, 0.306],
    [0.079, 0.334, 0.028, 0.09, 0.431, 0.0, 0.063, 0.022, 0.425, 0.122],
    [0.231, 0.554, 0.826, 0.776, 0.536, 0.063, 0.0, 0.806, 0.494, 0.245],
    [0.207, 0.521, 0.899, 0.786, 0.476, 0.022, 0.806, 0.0, 0.429, 0.219],
    [0.285, 0.686, 0.444, 0.456, 0.714, 0.425, 0.494, 0.429, 0.0, 0.279],
    [0.129, 0.291, 0.239, 0.22, 0.306, 0.122, 0.245, 0.219, 0.279, 0.0]
]) 

spearman_matrix_wild = np.array([
    [0.0, 0.182, 0.083, 0.123, 0.225, 0.155, 0.134, 0.13, 0.274, 0.141],
    [0.182, 0.0, 0.426, 0.484, 0.806, 0.713, 0.526, 0.362, 0.798, 0.34],
    [0.083, 0.426, 0.0, 0.748, 0.344, 0.311, 0.64, 0.757, 0.387, 0.151],
    [0.123, 0.484, 0.748, 0.0, 0.407, 0.371, 0.684, 0.649, 0.467, 0.191],
    [0.225, 0.806, 0.344, 0.407, 0.0, 0.736, 0.468, 0.302, 0.807, 0.359],
    [0.155, 0.713, 0.311, 0.371, 0.736, 0.0, 0.408, 0.245, 0.701, 0.276],
    [0.134, 0.526, 0.64, 0.684, 0.468, 0.408, 0.0, 0.578, 0.499, 0.219],
    [0.13, 0.362, 0.757, 0.649, 0.302, 0.245, 0.578, 0.0, 0.357, 0.157],
    [0.274, 0.798, 0.387, 0.467, 0.807, 0.701, 0.499, 0.357, 0.0, 0.414],
    [0.141, 0.34, 0.151, 0.191, 0.359, 0.276, 0.219, 0.157, 0.414, 0.0]
])

triangle_results = np.array([
    np.triu(spearman_matrix_mteb),
    np.triu(spearman_matrix_dice),
    np.triu(spearman_matrix_jbb),
    np.triu(spearman_matrix_prism),
    np.triu(spearman_matrix_sg),
    np.triu(spearman_matrix_wild),
    np.triu(spearman_matrix_gen),
    np.triu(spearman_matrix_safe),
    np.triu(spearman_matrix_gest)
])

num_datasets = 9
num_models = 10
flattened_matrices = triangle_results.reshape(num_datasets, -1)


dataset_avg = np.mean(triangle_results, axis=0) 


from scipy.cluster.hierarchy import dendrogram, linkage


# Function to generate dendrogram, average graph, elbow graph, k-means graph and rank graph
#
# Parameters: NONE
#
# Returns: NONE
def dendroGraph():
    # Calculate pairwise correlation distance between models
    # Use 1 - correlation as the distance metric
    corr_dist = 1 - dataset_avg

    # Perform hierarchical clustering
    Z = linkage(corr_dist, 'ward')

    # Plot dendrogram
    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=models)
    plt.title("Dendrogram of Models Based on Correlation")
    plt.xlabel("Models")
    plt.ylabel("Distance")
    plt.show()

# Function to generate average correlation across models
#
# Parameters: NONE
#
# Returns: NONE
def avgGraph():
    # Create a heatmap to visualize the average correlation across models
    plt.figure(figsize=(10, 8))
    sns.heatmap(dataset_avg, annot=True, cmap="coolwarm", linewidths=0.5,
                xticklabels=models,
                yticklabels=models)
    plt.title("Average Spearman Correlation Across Datasets")
    plt.xlabel("Models")
    plt.ylabel("Models")
    plt.show()

# Function to generate elbow graph for optimal K selection
#
# Parameters: NONE
#
# Returns: NONE
def elbowGraph():
    inertia_values = []
    k_values = range(1, 10)  # Experimenting with k from 1 to 9

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(flattened_matrices)
        inertia_values.append(kmeans.inertia_)

    # Plot the elbow graph
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertia_values, marker='o', linestyle='--')
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal K Selection")
    plt.xticks(k_values)
    plt.grid()
    plt.show()

# Function to generate K-Means clustering graph
#
# Parameters: NONE
#
# Returns: NONE
def kGraph():
    # Apply K-Means clustering
    k = 4  # Choose number of clusters based on optimal point from elbow graph
    kmeans = KMeans(n_clusters=k, random_state=42)
    dataset_clusters = kmeans.fit_predict(flattened_matrices)
    nameData = ['MTEB Dataset','DICE Dataset','JBB Dataset','PRISM Dataset','SG Dataset','WILD Dataset','GEN Dataset', 'SAFE Dataset','GEST Dataset']

    # Print cluster assignments
    for i, cluster in enumerate(dataset_clusters):
        print(f"{nameData[i]} -> Cluster {cluster}")

    # Plot clustering results
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(dataset_clusters)), dataset_clusters, c=dataset_clusters, cmap='viridis', marker='o')

    plt.xticks(ticks=range(len(nameData)), labels=nameData, rotation=45)

    plt.xlabel("Model")
    plt.ylabel("Cluster")
    plt.title("K-Means Clustering of Datasets Based on Correlations")
    plt.tight_layout()
    plt.show()

# Function to generate rank graph for model stability
#
# Parameters: NONE
#
#
# Returns: NONE
def rankGraph():
    rank_matrices = np.argsort(-triangle_results, axis=2)  # Get rank order for each dataset
    rank_std = np.std(rank_matrices, axis=0)  # Standard deviation of ranks across datasets

    plt.figure(figsize=(10, 6))
    sns.heatmap(rank_std, annot=True, cmap="coolwarm", xticklabels=models, yticklabels=models)
    plt.title("Rank Instability Across Datasets")
    plt.xlabel("Models")
    plt.ylabel("Models")
    plt.show()


# Generates a word cloud for each cluster text file
#
# Parameters:
# filePath: str - path to the text file containing the data
#
# Returns: NONE 
# 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def wordCloud(filePath):
    stopwords = STOPWORDS.union({ 
    "i", "you", "to", "a", "it", "of", "s", "that", "is", 
    "t", "can", "m", "and", "do", "be", "but", "not", "if", "in", 
    "are", "they", "like", "have", "me", "she", "my", "for", "what", 
    "just", "your", "about", "so", "with", "on", "her", "all", "re", 'go',
    "no", "when", "we", "think", "one", "need", "there", "this", 'got', 'let'
    "know", "them", "help", "the", "this", "how", "not", 'want', 'wants', 'will'
})
    
    with open(filePath, 'r', encoding='ISO-8859-1') as file:
        text = file.read()

    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)
            
    word_counts = {word: count for word, count in word_counts.items() if word not in stopwords}

    most_common_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate_from_frequencies(dict(most_common_words))

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show() 


import argparse

# Main function to run the script, parse arguments and call the run function 
def main():
    parser = argparse.ArgumentParser(description='Generate clusters from datasets')
    
    parser.add_argument('--input', type=str, required=True, help='Path to the input text file')
    args = parser.parse_args()

    fileName = 'ManageAgglo/agglo/agglomeration_results_safe/albert/cluster_5.txt'
    fileName = args.input
    wordCloud(fileName)
    dendroGraph()
    avgGraph()
    elbowGraph()
    kGraph()
    rankGraph()


if __name__ == "__main__":
    main()
