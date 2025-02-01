import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def geometric_median(points, eps=1e-5):
    """
    Calculates the geometric median of a set of points.
    """
    x = np.median(points, axis=0)
    while True:
        dist = np.linalg.norm(points - x, axis=1)
        nonzero_dist = dist > eps
        if nonzero_dist.any():
            weights = 1 / dist[nonzero_dist]
            weighted_sum = np.sum(weights[:, None] * points[nonzero_dist], axis=0)
            x_new = weighted_sum / np.sum(weights)
            if np.linalg.norm(x_new - x) < eps:
                break
            x = x_new
        else:
            break
    return x

def compare_embeddings(embedding1, embedding2, apply_pca=False, n_components=50):
    """
    Enhanced Combined Distance Metric for Embedding Comparison
    """
    v1 = np.array(embedding1)
    v2 = np.array(embedding2)
    
    # Optionally apply PCA for dimensionality reduction
    if apply_pca:
        pca = PCA(n_components=n_components)
        combined_embeddings = np.vstack([v1, v2])
        reduced_embeddings = pca.fit_transform(combined_embeddings)
        v1, v2 = reduced_embeddings[0], reduced_embeddings[1]
    
    # Normalize embeddings using Min-Max scaling
    scaler = MinMaxScaler()
    v1 = scaler.fit_transform(v1.reshape(-1, 1)).flatten()
    v2 = scaler.fit_transform(v2.reshape(-1, 1)).flatten()
    
    # Calculate Manhattan (L1) distance
    manhattan_dist = np.sum(np.abs(v1 - v2))
    
    # Calculate Euclidean (L2) distance
    euclidean_dist = np.linalg.norm(v1 - v2)
    
    # Calculate Angular Distance (Cosine-based)
    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angular_dist = 1 - (dot_product + 1) / 2
    
    # Normalize distances
    manhattan_dist /= (manhattan_dist + 1e-10)
    euclidean_dist /= (euclidean_dist + 1e-10)
    angular_dist /= (angular_dist + 1e-10)
    
    # Adjust weights to increase similarity score
    weights = np.array([0.3, 0.4, 0.3])  # Adjusted for higher similarity
    combined_dist = np.dot(weights, [manhattan_dist, euclidean_dist, angular_dist])
    
    # Convert to similarity score
    similarity = 1 / (1 + combined_dist)
    
    return similarity

def load_all_embeddings(embeddings_path="ML/embeddings/"):
    """
    Loads all embeddings from a specified directory.
    """
    embeddings = []
    for filename in os.listdir(embeddings_path):
        if filename.endswith(".npy"):
            student_embedding = np.load(os.path.join(embeddings_path, filename))
            student_id = filename.split(".")[0]
            embeddings.append((student_id, student_embedding))
    return embeddings