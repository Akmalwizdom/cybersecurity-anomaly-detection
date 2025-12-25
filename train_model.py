"""
=============================================================================
Cybersecurity Anomaly Detection Pipeline
K-Means Clustering 
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
import json
import os

os.makedirs('models', exist_ok=True)

print("="*70)
print(" CYBERSECURITY ANOMALY DETECTION")
print(" K-Means Clustering")
print("="*70)

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n" + "="*70)
print(" STEP 1: DATA LOADING")
print("="*70)

df = pd.read_csv('cybersecurity_attacks.csv')
print(f"✓ Dataset loaded: {df.shape[0]:,} records")

# =============================================================================
# 2. FEATURE SELECTION (OPTIMIZED)
# =============================================================================
print("\n" + "="*70)
print(" STEP 2: FEATURE SELECTION")
print("="*70)

# Use only Anomaly Scores - gives best Silhouette Score
print("\n📋 Using optimized feature set:")
print("   - Anomaly Scores (primary discriminator)")

# Prepare features
X = df[['Anomaly Scores']].copy()
X = X.fillna(X.median())

print(f"\n📊 Feature Statistics:")
print(f"   Mean: {X['Anomaly Scores'].mean():.2f}")
print(f"   Std:  {X['Anomaly Scores'].std():.2f}")
print(f"   Min:  {X['Anomaly Scores'].min():.2f}")
print(f"   Max:  {X['Anomaly Scores'].max():.2f}")

# =============================================================================
# 3. PREPROCESSING
# =============================================================================
print("\n" + "="*70)
print(" STEP 3: PREPROCESSING")
print("="*70)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"✓ StandardScaler applied")

# =============================================================================
# 4. K-MEANS CLUSTERING (k=2 for best separation)
# =============================================================================
print("\n" + "="*70)
print(" STEP 4: K-MEANS CLUSTERING")
print("="*70)

# Test different k values
print("\n📊 Evaluating k values:")
print("-" * 50)
print(f"{'k':<5} {'Silhouette':<15} {'Davies-Bouldin':<15}")
print("-" * 50)

best_k = 2
best_sil = 0

for k in range(2, 6):
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels_temp = kmeans_temp.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels_temp)
    db = davies_bouldin_score(X_scaled, labels_temp)
    
    marker = "🏆" if sil > best_sil else ""
    print(f"{k:<5} {sil:<15.4f} {db:<15.4f} {marker}")
    
    if sil > best_sil:
        best_sil = sil
        best_k = k

print("-" * 50)
print(f"\n🎯 Optimal k = {best_k} (Silhouette: {best_sil:.4f})")

# Train final model with best k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=20)
labels = kmeans.fit_predict(X_scaled)
df['Cluster'] = labels

# Final metrics
silhouette = silhouette_score(X_scaled, labels)
db_score = davies_bouldin_score(X_scaled, labels)

print(f"\n📊 Final Model Metrics:")
print(f"   - Silhouette Score: {silhouette:.4f}")
print(f"   - Davies-Bouldin Index: {db_score:.4f}")

# Cluster distribution
print(f"\n📈 Cluster Distribution:")
for cluster in range(best_k):
    count = (labels == cluster).sum()
    pct = count / len(labels) * 100
    print(f"   Cluster {cluster}: {count:,} ({pct:.1f}%)")

# =============================================================================
# 5. CLUSTER INTERPRETATION
# =============================================================================
print("\n" + "="*70)
print(" STEP 5: CLUSTER INTERPRETATION")
print("="*70)

cluster_stats = df.groupby('Cluster')['Anomaly Scores'].agg(['mean', 'std', 'min', 'max'])
print("\n📊 Cluster Statistics (Anomaly Scores):")
print(cluster_stats.round(2))

# Determine risk levels
cluster_interpretation = {}
for cluster in range(best_k):
    avg_score = df[df['Cluster'] == cluster]['Anomaly Scores'].mean()
    
    if avg_score >= 50:
        interpretation = {
            'risk_level': 'HIGH',
            'label': 'High Risk Traffic',
            'description': 'High anomaly scores - potential attack or suspicious activity',
            'color': '#ff4444'
        }
    else:
        interpretation = {
            'risk_level': 'LOW',
            'label': 'Normal Traffic',
            'description': 'Low anomaly scores - normal network traffic pattern',
            'color': '#00ff88'
        }
    
    interpretation['avg_anomaly_score'] = float(avg_score)
    cluster_interpretation[cluster] = interpretation
    
    print(f"\n   Cluster {cluster}:")
    print(f"   - Avg Anomaly Score: {avg_score:.2f}")
    print(f"   - Risk Level: {interpretation['risk_level']}")
    print(f"   - Label: {interpretation['label']}")

# =============================================================================
# 6. SAVE MODELS
# =============================================================================
print("\n" + "="*70)
print(" STEP 6: SAVE MODELS")
print("="*70)

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("✓ Saved: models/scaler.pkl")

# Save K-Means model
joblib.dump(kmeans, 'models/kmeans_model.pkl')
print("✓ Saved: models/kmeans_model.pkl")

# Save cluster interpretation
with open('models/cluster_interpretation.json', 'w') as f:
    json.dump(cluster_interpretation, f, indent=2)
print("✓ Saved: models/cluster_interpretation.json")

# Save model info
feature_info = {
    'features': ['Anomaly Scores'],
    'optimal_k': best_k,
    'silhouette_score': float(silhouette),
    'davies_bouldin_score': float(db_score),
    'model_type': 'KMeans',
    'scaler': 'StandardScaler'
}

with open('models/feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)
print("✓ Saved: models/feature_info.json")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print(" TRAINING COMPLETE")
print("="*70)

print(f"""
🎯 MODEL SUMMARY:
   - Algorithm: K-Means
   - Clusters: {best_k}
   - Features: Anomaly Scores only
   
📊 QUALITY METRICS:
   - Silhouette Score: {silhouette:.4f}
   - Davies-Bouldin Index: {db_score:.4f}
   
📁 SAVED FILES:
   - models/scaler.pkl
   - models/kmeans_model.pkl
   - models/cluster_interpretation.json
   - models/feature_info.json

📈 To generate visualizations, run:
   python create_visualizations.py

🚀 To start the app, run:
   python app.py
""")

print("="*70)
