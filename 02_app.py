import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("="*60)
print("LOADING DATASET")
print("="*60)

# Load the CSV file
df = pd.read_csv('spotify_tracks.csv')  # Replace with your actual file path

print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nColumn Names:")
print(df.columns.tolist())
print(f"\nData Info:")
print(df.info())

# ============================================================
# STEP 2: DATA PREPROCESSING
# ============================================================
print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

# Check for missing values
print(f"\nMissing Values:\n{df.isnull().sum()}")

# Handle missing values
df = df.dropna()

# Select only numeric audio features (DNA patterns)
audio_features = ['acousticness', 'danceability', 'duration_ms', 'energy', 
                  'instrumentalness', 'liveness', 'loudness', 'speechiness', 
                  'tempo', 'valence', 'key', 'mode', 'time_signature', 'popularity', 'year']

# Create a copy with only audio features
X = df[audio_features].copy()

print(f"\nAudio Features Selected: {len(audio_features)} features")
print(audio_features)

# Basic statistics
print(f"\nBasic Statistics of Audio Features:")
print(X.describe())

# ============================================================
# STEP 3: FEATURE SCALING
# ============================================================
print("\n" + "="*60)
print("FEATURE SCALING")
print("="*60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Features scaled using StandardScaler")
print(f"Scaled data shape: {X_scaled.shape}")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved as 'scaler.pkl'")

# ============================================================
# STEP 4: DIMENSIONALITY REDUCTION USING PCA
# ============================================================
print("\n" + "="*60)
print("DIMENSIONALITY REDUCTION (PCA)")
print("="*60)

# Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X_scaled)

print(f"Original number of features: {X_scaled.shape[1]}")
print(f"Reduced number of features: {X_pca.shape[1]}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")

# Visualize explained variance
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('Variance Explained by Each PC')

plt.subplot(1, 2, 2)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.legend()
plt.tight_layout()
plt.savefig('pca_analysis.png')
print("PCA visualization saved as 'pca_analysis.png'")

# Save PCA
with open('pca.pkl', 'wb') as f:
    pickle.dump(pca, f)
print("PCA model saved as 'pca.pkl'")

# ============================================================
# STEP 5: CLUSTERING MODEL 1 - K-MEANS
# ============================================================
print("\n" + "="*60)
print("MODEL 1: K-MEANS CLUSTERING")
print("="*60)

# Find optimal number of clusters using elbow method
inertias = []
silhouette_scores = []
K_range = range(5, 21)

print("\nFinding optimal number of clusters...")
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_pca, kmeans.labels_))
    print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.4f}")

# Plot elbow curve
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, marker='o', color='green')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score Analysis')
plt.tight_layout()
plt.savefig('kmeans_optimization.png')
print("K-Means optimization plot saved as 'kmeans_optimization.png'")

# Train final K-Means model with optimal K
optimal_k = 10  # You can adjust based on the plots
kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans_model.fit_predict(X_pca)

# Evaluate K-Means
kmeans_silhouette = silhouette_score(X_pca, kmeans_labels)
kmeans_db = davies_bouldin_score(X_pca, kmeans_labels)
kmeans_ch = calinski_harabasz_score(X_pca, kmeans_labels)

print(f"\nK-Means Model (K={optimal_k}) Performance:")
print(f"Silhouette Score: {kmeans_silhouette:.4f} (Higher is better, range: -1 to 1)")
print(f"Davies-Bouldin Index: {kmeans_db:.4f} (Lower is better)")
print(f"Calinski-Harabasz Score: {kmeans_ch:.2f} (Higher is better)")

# Save K-Means model
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans_model, f)
print("K-Means model saved as 'kmeans_model.pkl'")

# ============================================================
# STEP 6: CLUSTERING MODEL 2 - DBSCAN
# ============================================================
print("\n" + "="*60)
print("MODEL 2: DBSCAN CLUSTERING")
print("="*60)

# Train DBSCAN
dbscan_model = DBSCAN(eps=2.5, min_samples=5)
dbscan_labels = dbscan_model.fit_predict(X_pca)

# Count clusters
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"\nDBSCAN Results:")
print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

# Evaluate DBSCAN (only if we have more than 1 cluster)
if n_clusters > 1:
    # Exclude noise points for evaluation
    mask = dbscan_labels != -1
    if sum(mask) > 0:
        dbscan_silhouette = silhouette_score(X_pca[mask], dbscan_labels[mask])
        dbscan_db = davies_bouldin_score(X_pca[mask], dbscan_labels[mask])
        dbscan_ch = calinski_harabasz_score(X_pca[mask], dbscan_labels[mask])
        
        print(f"\nDBSCAN Performance:")
        print(f"Silhouette Score: {dbscan_silhouette:.4f} (Higher is better)")
        print(f"Davies-Bouldin Index: {dbscan_db:.4f} (Lower is better)")
        print(f"Calinski-Harabasz Score: {dbscan_ch:.2f} (Higher is better)")
    else:
        print("Not enough valid clusters for evaluation")
        dbscan_silhouette = dbscan_db = dbscan_ch = 0
else:
    print("DBSCAN found insufficient clusters for evaluation")
    dbscan_silhouette = dbscan_db = dbscan_ch = 0

# Save DBSCAN model
with open('dbscan_model.pkl', 'wb') as f:
    pickle.dump(dbscan_model, f)
print("DBSCAN model saved as 'dbscan_model.pkl'")

# ============================================================
# STEP 7: CLUSTERING MODEL 3 - HIERARCHICAL CLUSTERING
# ============================================================
print("\n" + "="*60)
print("MODEL 3: HIERARCHICAL CLUSTERING")
print("="*60)

# Check dataset size and sample if too large
max_samples_hierarchical = 10000  # Limit for memory safety

if len(X_pca) > max_samples_hierarchical:
    print(f"Dataset has {len(X_pca)} samples. Sampling {max_samples_hierarchical} for Hierarchical clustering to prevent memory issues...")
    
    # Random sampling for training
    sample_indices = np.random.choice(len(X_pca), max_samples_hierarchical, replace=False)
    X_pca_sample = X_pca[sample_indices]
    
    # Train on sample
    hierarchical_model = AgglomerativeClustering(n_clusters=optimal_k)
    hierarchical_labels_sample = hierarchical_model.fit_predict(X_pca_sample)
    
    # For full dataset, use KMeans on the centroids approach as approximation
    from sklearn.neighbors import KNeighborsClassifier
    
    # Create a KNN classifier to extend labels to full dataset
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_pca_sample, hierarchical_labels_sample)
    hierarchical_labels = knn.predict(X_pca)
    
    # Evaluate on sample
    hierarchical_silhouette = silhouette_score(X_pca_sample, hierarchical_labels_sample)
    hierarchical_db = davies_bouldin_score(X_pca_sample, hierarchical_labels_sample)
    hierarchical_ch = calinski_harabasz_score(X_pca_sample, hierarchical_labels_sample)
    
    print(f"Note: Trained on {max_samples_hierarchical} samples, extended to full dataset using KNN")
    
else:
    # Train Hierarchical Clustering on full dataset
    hierarchical_model = AgglomerativeClustering(n_clusters=optimal_k)
    hierarchical_labels = hierarchical_model.fit_predict(X_pca)
    
    # Evaluate Hierarchical
    hierarchical_silhouette = silhouette_score(X_pca, hierarchical_labels)
    hierarchical_db = davies_bouldin_score(X_pca, hierarchical_labels)
    hierarchical_ch = calinski_harabasz_score(X_pca, hierarchical_labels)

print(f"\nHierarchical Clustering (n_clusters={optimal_k}) Performance:")
print(f"Silhouette Score: {hierarchical_silhouette:.4f} (Higher is better)")
print(f"Davies-Bouldin Index: {hierarchical_db:.4f} (Lower is better)")
print(f"Calinski-Harabasz Score: {hierarchical_ch:.2f} (Higher is better)")

# Save Hierarchical model (note: cannot predict on new data directly)
# We save the labels instead
hierarchical_data = {
    'labels': hierarchical_labels,
    'n_clusters': optimal_k
}
with open('hierarchical_model.pkl', 'wb') as f:
    pickle.dump(hierarchical_data, f)
print("Hierarchical clustering labels saved as 'hierarchical_model.pkl'")

# ============================================================
# STEP 8: COMPARISON OF MODELS
# ============================================================
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

comparison_df = pd.DataFrame({
    'Model': ['K-Means', 'DBSCAN', 'Hierarchical'],
    'Silhouette Score': [kmeans_silhouette, dbscan_silhouette, hierarchical_silhouette],
    'Davies-Bouldin Index': [kmeans_db, dbscan_db, hierarchical_db],
    'Calinski-Harabasz Score': [kmeans_ch, dbscan_ch, hierarchical_ch]
})

print("\n", comparison_df.to_string(index=False))

# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].bar(comparison_df['Model'], comparison_df['Silhouette Score'], color=['blue', 'green', 'orange'])
axes[0].set_ylabel('Silhouette Score')
axes[0].set_title('Silhouette Score (Higher is Better)')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(comparison_df['Model'], comparison_df['Davies-Bouldin Index'], color=['blue', 'green', 'orange'])
axes[1].set_ylabel('Davies-Bouldin Index')
axes[1].set_title('Davies-Bouldin Index (Lower is Better)')
axes[1].tick_params(axis='x', rotation=45)

axes[2].bar(comparison_df['Model'], comparison_df['Calinski-Harabasz Score'], color=['blue', 'green', 'orange'])
axes[2].set_ylabel('Calinski-Harabasz Score')
axes[2].set_title('Calinski-Harabasz Score (Higher is Better)')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model_comparison.png')
print("\nModel comparison plot saved as 'model_comparison.png'")

# ============================================================
# STEP 9: SAVE PREPROCESSED DATA WITH CLUSTER LABELS
# ============================================================
print("\n" + "="*60)
print("SAVING PROCESSED DATA")
print("="*60)

# Add cluster labels to original dataframe
df['kmeans_cluster'] = kmeans_labels
df['dbscan_cluster'] = dbscan_labels
df['hierarchical_cluster'] = hierarchical_labels

# Save processed data
df.to_csv('processed_music_data.csv', index=False)
print("Processed data with cluster labels saved as 'processed_music_data.csv'")

# Save feature data
np.save('X_pca.npy', X_pca)
print("PCA-transformed features saved as 'X_pca.npy'")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print("\nFiles created:")
print("1. scaler.pkl - Feature scaler")
print("2. pca.pkl - PCA model")
print("3. kmeans_model.pkl - K-Means clustering model")
print("4. dbscan_model.pkl - DBSCAN clustering model")
print("5. hierarchical_model.pkl - Hierarchical clustering model")
print("6. processed_music_data.csv - Dataset with cluster labels")
print("7. X_pca.npy - PCA-transformed features")
print("8. Various .png visualization files")
print("\nYou can now run the Streamlit application!")