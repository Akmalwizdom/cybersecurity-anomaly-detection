"""
=============================================================================
Flask Backend - Threat Intelligence Dashboard
Cybersecurity Anomaly Detection API
=============================================================================
"""

from flask import Flask, render_template, request, jsonify
import joblib
import json
import numpy as np
import os

app = Flask(__name__)

# ============================================================================
# MODEL LOADING
# ============================================================================

MODEL_DIR = 'models'

try:
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    kmeans = joblib.load(os.path.join(MODEL_DIR, 'kmeans_model.pkl'))
    
    with open(os.path.join(MODEL_DIR, 'cluster_interpretation.json'), 'r') as f:
        cluster_interpretation = json.load(f)
    
    with open(os.path.join(MODEL_DIR, 'feature_info.json'), 'r') as f:
        feature_info = json.load(f)
    
    print("✓ All models loaded successfully!")
    MODELS_LOADED = True
except Exception as e:
    print(f"⚠ Warning: Could not load models - {e}")
    print("  Please run train_model.py first to train the model.")
    MODELS_LOADED = False

# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Halaman utama - Threat Intelligence Dashboard"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Endpoint untuk menganalisis traffic dan memprediksi cluster.
    
    Expected JSON payload:
    {
        "anomaly_scores": float (0-100)
    }
    
    Optional fields (for display only):
    {
        "packet_length": int,
        "protocol": str,
        "network_segment": str
    }
    """
    if not MODELS_LOADED:
        return jsonify({
            'success': False,
            'error': 'Models not loaded. Please run train_model.py first.'
        }), 500
    
    try:
        data = request.get_json()
        
        # Get anomaly scores (required)
        if 'anomaly_scores' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: anomaly_scores'
            }), 400
        
        anomaly_scores = float(data['anomaly_scores'])
        
        # Validate range
        if not (0 <= anomaly_scores <= 100):
            return jsonify({
                'success': False,
                'error': 'Anomaly scores must be between 0 and 100'
            }), 400
        
        # Get optional fields for display
        packet_length = data.get('packet_length', 500)
        protocol = data.get('protocol', 'TCP')
        network_segment = data.get('network_segment', 'Segment A')
        
        # Prepare input for model (only Anomaly Scores)
        X = np.array([[anomaly_scores]])
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        cluster = int(kmeans.predict(X_scaled)[0])
        
        # Calculate distance to centroid
        centroid = kmeans.cluster_centers_[cluster]
        distance = float(np.abs(X_scaled[0][0] - centroid[0]))
        
        # Get interpretation
        interpretation = cluster_interpretation.get(str(cluster), {
            'risk_level': 'UNKNOWN',
            'label': 'Unknown',
            'description': 'Unable to determine risk level',
            'color': '#888888'
        })
        
        # Response
        response = {
            'success': True,
            'cluster': cluster,
            'distance_to_centroid': round(distance, 4),
            'interpretation': interpretation,
            'input_data': {
                'anomaly_scores': anomaly_scores,
                'packet_length': packet_length,
                'protocol': protocol,
                'network_segment': network_segment
            }
        }
        
        return jsonify(response)
    
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid value: {str(e)}'
        }), 400
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}'
        }), 500


@app.route('/api/info')
def api_info():
    """Endpoint untuk mendapatkan informasi model"""
    if not MODELS_LOADED:
        return jsonify({
            'success': False,
            'error': 'Models not loaded'
        }), 500
    
    return jsonify({
        'success': True,
        'model_info': feature_info,
        'cluster_count': len(cluster_interpretation),
        'clusters': cluster_interpretation
    })


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print(" THREAT INTELLIGENCE DASHBOARD")
    print(" Cybersecurity Anomaly Detection System")
    print("="*60)
    print("\n🌐 Server starting...")
    print("   URL: http://localhost:5000")
    print("\n📊 Endpoints:")
    print("   GET  /          - Dashboard (Web UI)")
    print("   POST /analyze   - Analyze traffic")
    print("   GET  /api/info  - Model information")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
