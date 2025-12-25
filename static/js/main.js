// Update timestamp
function updateTimestamp() {
    const now = new Date();
    const timestamp = now.toISOString().replace('T', ' ').substring(0, 19);
    document.getElementById('timestamp').textContent = timestamp + ' UTC';
}
updateTimestamp();
setInterval(updateTimestamp, 1000);

// Console log function
function log(message, type = 'info') {
    const consoleLog = document.getElementById('consoleLog');
    const className = type === 'error' ? 'text-red-400' : type === 'success' ? 'text-green-400' : 'text-gray-400';
    const prefix = type === 'error' ? '✗' : type === 'success' ? '✓' : '>';
    consoleLog.innerHTML += `<p class="${className}">${prefix} ${message}</p>`;
    consoleLog.scrollTop = consoleLog.scrollHeight;
}

// Form submission
document.getElementById('analyzeForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const submitBtn = document.getElementById('submitBtn');
    const btnText = document.getElementById('btnText');
    const btnLoader = document.getElementById('btnLoader');
    
    // Disable button and show loader
    submitBtn.disabled = true;
    btnText.textContent = 'MENGANALISIS...';
    btnLoader.classList.remove('hidden');
    
    log('Memulai analisis trafik jaringan...');
    
    const formData = {
        anomaly_scores: parseFloat(document.getElementById('anomaly_scores').value),
        packet_length: parseInt(document.getElementById('packet_length').value),
        protocol: document.getElementById('protocol').value,
        network_segment: document.getElementById('network_segment').value
    };
    
    log(`Parameter: AS=${formData.anomaly_scores}, PL=${formData.packet_length}`);
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            log('Analisis selesai dengan sukses', 'success');
            displayResults(data);
        } else {
            log(`Error: ${data.error}`, 'error');
            alert('Analysis failed: ' + data.error);
        }
    } catch (error) {
        log(`Request failed: ${error.message}`, 'error');
        alert('Request failed: ' + error.message);
    } finally {
        submitBtn.disabled = false;
        btnText.textContent = 'MULAI ANALISIS';
        btnLoader.classList.add('hidden');
    }
});

// Display results
function displayResults(data) {
    document.getElementById('noResults').classList.add('hidden');
    document.getElementById('resultsContainer').classList.remove('hidden');
    
    const interp = data.interpretation;
    const riskLevel = interp.risk_level.toLowerCase();
    
    // Update cluster card
    const clusterCard = document.getElementById('clusterCard');
    clusterCard.className = `border rounded-xl p-6 text-center transition-all duration-300 risk-bg-${riskLevel}`;
    
    document.getElementById('clusterID').textContent = data.cluster;
    
    const riskLabelEl = document.getElementById('riskLabel');
    riskLabelEl.textContent = interp.label;
    riskLabelEl.className = `text-lg font-medium risk-${riskLevel}`;
    
    // Update risk badge
    const riskBadge = document.getElementById('riskBadge');
    riskBadge.textContent = interp.risk_level;
    riskBadge.className = `text-xs px-2 py-1 rounded-full border risk-${riskLevel}`;
    
    // Update risk bar
    const riskBar = document.getElementById('riskBar');
    let barWidth = riskLevel === 'high' ? 100 : riskLevel === 'medium' ? 60 : 30;
    let barColor = riskLevel === 'high' ? '#ff4444' : riskLevel === 'medium' ? '#ffaa00' : '#00ff88';
    riskBar.style.setProperty('--bar-width', barWidth + '%');
    riskBar.style.width = barWidth + '%';
    riskBar.style.backgroundColor = barColor;
    
    // Update description
    document.getElementById('description').textContent = interp.description;
    
    // Update metrics
    document.getElementById('distanceValue').textContent = data.distance_to_centroid !== undefined ? data.distance_to_centroid.toFixed(4) : 'N/A';
    document.getElementById('avgScore').textContent = interp.avg_anomaly_score !== undefined ? interp.avg_anomaly_score.toFixed(2) : 'N/A';
    
    // Update raw data
    document.getElementById('rawData').textContent = JSON.stringify(data.input_data, null, 2);
    
    log(`Cluster ${data.cluster} terdeteksi: RISIKO ${interp.risk_level}`, 
        riskLevel === 'high' ? 'error' : riskLevel === 'medium' ? 'info' : 'success');
}
