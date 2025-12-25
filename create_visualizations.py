"""
=============================================================================
Cybersecurity Attack Clustering - Visualization Script
Creates 5 high-quality visualizations for cluster analysis
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# =============================================================================
# CONFIGURATION
# =============================================================================
plt.style.use('dark_background')
os.makedirs('visualizations', exist_ok=True)

# Neon colors for cybersecurity theme
NEON_GREEN = '#00ff88'
NEON_RED = '#ff4444'
COLORS = [NEON_GREEN, NEON_RED, '#ffaa00', '#00aaff', '#ff00ff']

OPTIMAL_K = 2
DPI = 300

print("="*70)
print(" CYBERSECURITY CLUSTERING - VISUALIZATION GENERATOR")
print("="*70)

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n[1/5] Loading data...")

CSV_FILE = 'cybersecurity_attacks.csv'
if not os.path.exists(CSV_FILE):
    print(f"❌ Error: File '{CSV_FILE}' not found!")
    sys.exit(1)

df = pd.read_csv(CSV_FILE)
print(f"✓ Dataset loaded: {df.shape[0]:,} records, {df.shape[1]} columns")

# Identify numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"✓ Numeric columns found: {len(numeric_cols)}")

# Fill missing values with median
for col in numeric_cols:
    if df[col].isna().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

# Prepare Anomaly Scores for clustering
X_anomaly = df[['Anomaly Scores']].values

# StandardScaler for clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_anomaly)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=OPTIMAL_K, random_state=42, n_init=20)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Determine which cluster is "Attack" vs "Normal" based on mean Anomaly Score
cluster_means = df.groupby('Cluster')['Anomaly Scores'].mean()
attack_cluster = cluster_means.idxmax()
normal_cluster = cluster_means.idxmin()

# Create cluster label mapping
cluster_labels = {
    attack_cluster: 'Attack Traffic',
    normal_cluster: 'Normal Traffic'
}
df['Cluster_Label'] = df['Cluster'].map(cluster_labels)

print(f"✓ K-Means clustering complete (k={OPTIMAL_K})")
print(f"  - Normal Traffic (Cluster {normal_cluster}): {(df['Cluster']==normal_cluster).sum():,}")
print(f"  - Attack Traffic (Cluster {attack_cluster}): {(df['Cluster']==attack_cluster).sum():,}")

# =============================================================================
# 2. VISUALIZATION 1: SCATTER PLOT WITH CENTROIDS
# =============================================================================
print("\n[2/5] Creating Scatter Plot with Centroids...")

fig, ax = plt.subplots(figsize=(12, 8))

# Find a second numeric feature with high variance (excluding Anomaly Scores)
numeric_features = [col for col in numeric_cols if col != 'Anomaly Scores']

if len(numeric_features) > 0:
    # Calculate variance and pick highest
    variances = {col: df[col].var() for col in numeric_features}
    y_feature = max(variances, key=variances.get)
    y_data = df[y_feature].values
    y_label = y_feature
else:
    # Fallback: use index with small noise
    y_data = np.arange(len(df)) + np.random.normal(0, 0.1, len(df))
    y_label = 'Record Index'

# Plot scatter for each cluster
for cluster in [normal_cluster, attack_cluster]:
    mask = df['Cluster'] == cluster
    color = NEON_GREEN if cluster == normal_cluster else NEON_RED
    label = cluster_labels[cluster]
    ax.scatter(df.loc[mask, 'Anomaly Scores'], y_data[mask], 
               c=color, alpha=0.3, s=5, label=label)

# Calculate and plot centroids
centroids_x = []
centroids_y = []
for cluster in [normal_cluster, attack_cluster]:
    mask = df['Cluster'] == cluster
    cx = df.loc[mask, 'Anomaly Scores'].mean()
    cy = y_data[mask].mean()
    centroids_x.append(cx)
    centroids_y.append(cy)
    color = NEON_GREEN if cluster == normal_cluster else NEON_RED
    ax.scatter(cx, cy, c=color, marker='X', s=400, edgecolors='white', 
               linewidths=2, zorder=5)

ax.set_xlabel('Anomaly Scores', fontsize=12, fontweight='bold')
ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
ax.set_title('Scatter Plot with Cluster Centroids', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')

# Add centroid annotation
for i, (cx, cy) in enumerate(zip(centroids_x, centroids_y)):
    cluster = [normal_cluster, attack_cluster][i]
    ax.annotate(f'Centroid\n({cx:.1f}, {cy:.1f})', 
                xy=(cx, cy), xytext=(15, 15), textcoords='offset points',
                fontsize=9, color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='white', alpha=0.7))

plt.tight_layout()
plt.savefig('visualizations/scatter.png', dpi=DPI, bbox_inches='tight', facecolor='black')
plt.close()
print("✓ Saved: visualizations/scatter.png")

# =============================================================================
# 3. VISUALIZATION 2: PCA PROJECTION
# =============================================================================
print("\n[3/5] Creating PCA Projection...")

fig, ax = plt.subplots(figsize=(12, 8))

# Use all numeric features for PCA
X_all_numeric = df[numeric_cols].values

# Standardize before PCA
scaler_pca = StandardScaler()
X_pca_scaled = scaler_pca.fit_transform(X_all_numeric)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_pca_scaled)

# Plot PCA results
for cluster in [normal_cluster, attack_cluster]:
    mask = df['Cluster'] == cluster
    color = NEON_GREEN if cluster == normal_cluster else NEON_RED
    label = cluster_labels[cluster]
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               c=color, alpha=0.4, s=5, label=label)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', 
              fontsize=12, fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', 
              fontsize=12, fontweight='bold')
ax.set_title('PCA Dimensionality Reduction (All Numeric Features)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')

# Add variance info box
total_var = (pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1]) * 100
textstr = f'Total Variance Explained: {total_var:.1f}%\nFeatures Used: {len(numeric_cols)}'
props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7, edgecolor='white')
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, color='white')

plt.tight_layout()
plt.savefig('visualizations/pca.png', dpi=DPI, bbox_inches='tight', facecolor='black')
plt.close()
print("✓ Saved: visualizations/pca.png")

# =============================================================================
# 4. VISUALIZATION 3: ELBOW METHOD
# =============================================================================
print("\n[4/5] Creating Elbow Method Plot...")

fig, ax = plt.subplots(figsize=(10, 6))

k_range = range(1, 11)
inertias = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Plot line chart
ax.plot(k_range, inertias, 'o-', color=NEON_GREEN, linewidth=2, markersize=10, 
        markerfacecolor=NEON_GREEN, markeredgecolor='white', markeredgewidth=2)

# Highlight optimal k
ax.axvline(x=OPTIMAL_K, color=NEON_RED, linestyle='--', linewidth=2, 
           label=f'Optimal k = {OPTIMAL_K}')
ax.scatter([OPTIMAL_K], [inertias[OPTIMAL_K-1]], color=NEON_RED, s=200, 
           zorder=5, edgecolors='white', linewidths=2)

# Add inertia values
for i, (k, inertia) in enumerate(zip(k_range, inertias)):
    ax.annotate(f'{inertia:.0f}', xy=(k, inertia), xytext=(0, 10), 
                textcoords='offset points', ha='center', fontsize=8, color='white')

ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
ax.set_ylabel('Inertia (WCSS)', fontsize=12, fontweight='bold')
ax.set_title('Elbow Method - Optimal Number of Clusters', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(list(k_range))
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('visualizations/elbow.png', dpi=DPI, bbox_inches='tight', facecolor='black')
plt.close()
print("✓ Saved: visualizations/elbow.png")

# =============================================================================
# 5. VISUALIZATION 4: SILHOUETTE SCORE ANALYSIS
# =============================================================================
print("\n[5/5] Creating Silhouette Score Analysis...")

fig, ax = plt.subplots(figsize=(10, 6))

k_range_sil = range(2, 11)
sil_scores = []

for k in k_range_sil:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    sil_scores.append(sil)

# Color bars - highlight best score
max_sil = max(sil_scores)
bar_colors = [NEON_GREEN if s == max_sil else '#00aaff' for s in sil_scores]

bars = ax.bar(k_range_sil, sil_scores, color=bar_colors, edgecolor='white', linewidth=2)

# Add value labels on bars
for bar, score in zip(bars, sil_scores):
    height = bar.get_height()
    ax.annotate(f'{score:.4f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords='offset points',
                ha='center', fontsize=10, fontweight='bold', color='white')

# Add reference line
ax.axhline(y=0.5, color='#ffaa00', linestyle='--', linewidth=2, label='Good threshold (0.5)')

ax.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
ax.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
ax.set_title('Silhouette Score Analysis', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(list(k_range_sil))
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.set_ylim(0, max(sil_scores) * 1.2)

plt.tight_layout()
plt.savefig('visualizations/silhouette.png', dpi=DPI, bbox_inches='tight', facecolor='black')
plt.close()
print("✓ Saved: visualizations/silhouette.png")

# =============================================================================
# 6. VISUALIZATION 5: RADAR CHART (CLUSTER PROFILE)
# =============================================================================
print("\n[6/6] Creating Radar Chart (Cluster Profile)...")

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Select top numeric features (max 8 for readability), excluding Cluster column
radar_features = [col for col in numeric_cols if col not in ['Cluster']]
feature_variances = {col: df[col].var() for col in radar_features}
top_features = sorted(feature_variances.keys(), key=lambda x: feature_variances[x], reverse=True)[:8]

print(f"  - Using features: {top_features}")

# FIXED: Normalize the ENTIRE dataset first, then calculate cluster means
# This preserves the relative differences between clusters
df_normalized = df.copy()
minmax_scaler = MinMaxScaler()
df_normalized[top_features] = minmax_scaler.fit_transform(df[top_features])

# Now calculate means per cluster on normalized data
cluster_means_normalized = df_normalized.groupby('Cluster')[top_features].mean()

print(f"  - Normal Traffic means: {cluster_means_normalized.loc[normal_cluster].values.round(3)}")
print(f"  - Attack Traffic means: {cluster_means_normalized.loc[attack_cluster].values.round(3)}")

# Prepare radar chart
categories = top_features
N = len(categories)

# Compute angle for each category
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Complete the loop

# Plot for each cluster
for cluster in [normal_cluster, attack_cluster]:
    values = cluster_means_normalized.loc[cluster].values.tolist()
    values += values[:1]  # Complete the loop
    
    color = NEON_GREEN if cluster == normal_cluster else NEON_RED
    label = cluster_labels[cluster]
    
    ax.plot(angles, values, 'o-', linewidth=3, color=color, label=label, markersize=8)
    ax.fill(angles, values, alpha=0.25, color=color)

# Set labels with better formatting
ax.set_xticks(angles[:-1])
feature_labels = []
for f in categories:
    # Shorten long feature names
    if len(f) > 12:
        label = f.replace(' ', '\n').replace('_', '\n')
    else:
        label = f
    feature_labels.append(label)
ax.set_xticklabels(feature_labels, fontsize=9, fontweight='bold')

ax.set_title('Cluster Profile Comparison (Radar Chart)\nNormalized Feature Means (0-1 Scale)', 
             fontsize=14, fontweight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)

# Customize grid
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.yaxis.grid(True, color='white', alpha=0.3, linestyle='--')
ax.xaxis.grid(True, color='white', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('visualizations/radar.png', dpi=DPI, bbox_inches='tight', facecolor='black')
plt.close()
print("✓ Saved: visualizations/radar.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print(" VISUALIZATION COMPLETE")
print("="*70)
print(f"""
📊 GENERATED VISUALIZATIONS:
   1. scatter.png    - Scatter Plot with Centroids
   2. pca.png        - PCA Dimensionality Reduction
   3. elbow.png      - Elbow Method Analysis
   4. silhouette.png - Silhouette Score Analysis
   5. radar.png      - Cluster Profile Radar Chart

📁 Output folder: visualizations/
🎨 Style: Dark Background (Cybersecurity Theme)
🖼️ Resolution: {DPI} DPI (High-Res)

✅ All visualizations saved successfully!
""")
print("="*70)
