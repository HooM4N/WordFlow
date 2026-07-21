// DOM Elements
const folderInput = document.getElementById('folder-input');
const sysStatus = document.getElementById('sys-status');
const mainContent = document.getElementById('main-content');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');

// Inference Elements
const btnGenerate = document.getElementById('btn-generate');
const btnSimilar = document.getElementById('btn-similar');
const outStory = document.getElementById('out-story');
const outSimilar = document.getElementById('out-similar');

// Chart Registry
window.charts = {};

// Helpers
function showLoading(msg) {
    loadingText.innerText = msg;
    loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    loadingOverlay.classList.add('hidden');
}

function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(2) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num;
}

// Chart Initialization
function renderCharts(logs) {
    const epochs = logs.train_loss.map((_, i) => i + 1);

    // Destroy existing charts to prevent canvas memory leaks
    ['loss', 'ppl', 'lr'].forEach(k => {
        if (window.charts[k]) window.charts[k].destroy();
    });

    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { position: 'top', labels: { boxWidth: 10, font: { size: 11 } } }
        },
        scales: {
            x: { grid: { display: false }, ticks: { font: { size: 10 } } },
            y: { grid: { color: 'rgba(0,0,0,0.05)', drawBorder: false }, ticks: { font: { size: 10 } } }
        }
    };

    // 1. Loss Chart
    const ctxLoss = document.getElementById('chart-loss').getContext('2d');
    window.charts.loss = new Chart(ctxLoss, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                { label: 'Train', data: logs.train_loss, borderColor: '#0ea5e9', tension: 0.3, pointRadius: 0, borderWidth: 2 },
                { label: 'Val', data: logs.val_loss || [], borderColor: '#e11d48', tension: 0.3, pointRadius: 0, borderWidth: 2 }
            ]
        },
        options: commonOptions
    });

    // 2. Perplexity Chart
    const ctxPpl = document.getElementById('chart-ppl').getContext('2d');
    window.charts.ppl = new Chart(ctxPpl, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                { 
                    label: 'Perplexity', 
                    data: logs.val_perplexity || [], 
                    borderColor: '#10b981', 
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: true,
                    tension: 0.3, 
                    pointRadius: 0, 
                    borderWidth: 2 
                }
            ]
        },
        options: commonOptions
    });

    // 3. Learning Rate Chart
    const ctxLr = document.getElementById('chart-lr').getContext('2d');
    window.charts.lr = new Chart(ctxLr, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                { label: 'Learning Rate', data: logs.lr, borderColor: '#f59e0b', tension: 0.3, pointRadius: 0, borderWidth: 2 }
            ]
        },
        options: commonOptions
    });
}

// Native HTML5 File Folder Selection Flow
folderInput.addEventListener('change', async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    // Isolate the 4 required files from the selected folder
    let ckpt, logs, tokenizer, config;
    
    for (let f of files) {
        if (f.name === 'checkpoint_best.pt') ckpt = f;
        if (f.name === 'training_logs.json') logs = f;
        if (f.name === 'tokenizer.json') tokenizer = f;
        if (f.name === 'config.json') config = f;
    }

    if (!ckpt || !logs || !tokenizer || !config) {
        alert("Invalid directory selected.\nMissing one or more required files: checkpoint_best.pt, training_logs.json, tokenizer.json, config.json");
        e.target.value = ''; // Reset input
        return;
    }

    showLoading("Uploading & Loading Model...");

    // Prepare files for multipart/form-data upload
    const formData = new FormData();
    formData.append('checkpoint', ckpt);
    formData.append('logs', logs);
    formData.append('tokenizer', tokenizer);
    formData.append('config', config);

    try {
        const loadRes = await fetch('/api/load-run', {
            method: 'POST',
            body: formData
        });
        const loadData = await loadRes.json();
        
        if (!loadRes.ok) {
            throw new Error(loadData.detail || "Failed to load run.");
        }

        // Update UI Stats
        document.getElementById('stat-params').innerText = formatNumber(loadData.stats.params);
        document.getElementById('stat-emb').innerText = loadData.stats.emb_dim;
        document.getElementById('stat-hidden').innerText = loadData.stats.hidden_dim;
        document.getElementById('stat-layers').innerText = loadData.stats.layers;
        document.getElementById('stat-seq').innerText = loadData.stats.seq_len;
        document.getElementById('stat-epochs').innerText = loadData.stats.epochs;
        document.getElementById('stat-lr').innerText = loadData.stats.init_lr;
        document.getElementById('stat-wd').innerText = loadData.stats.weight_decay;
        document.getElementById('stat-time').innerText = loadData.stats.mean_epoch_time;

        // Render Plots
        renderCharts(loadData.logs);

        // Unlock Interface
        mainContent.classList.remove('disabled');
        sysStatus.innerText = `Active [${loadData.device.toUpperCase()}]`;
        sysStatus.classList.add('active');
        outStory.innerHTML = '<span class="empty-state">Parameters loaded. Ready to generate.</span>';
        outSimilar.innerHTML = '<span class="empty-state">Embeddings mapped. Ready to search.</span>';

    } catch (err) {
        alert("Error loading run:\n" + err.message);
    } finally {
        hideLoading();
        e.target.value = ''; // Reset input so user can pick the same folder again if needed
    }
});

// Inference: Generate Story
btnGenerate.addEventListener('click', async () => {
    const maxTokens = document.getElementById('inp-len').value || 150;
    const temp = document.getElementById('inp-temp').value || 0.8;
    const seed = document.getElementById('inp-seed').value;
    
    outStory.innerHTML = '<span class="empty-state">Generating autoregressively...</span>';
    btnGenerate.disabled = true;

    try {
        const res = await fetch('/api/generate', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                max_tokens: parseInt(maxTokens), 
                temperature: parseFloat(temp), 
                seed: seed ? parseInt(seed) : null
            })
        });
        const data = await res.json();
        
        if (!res.ok) throw new Error(data.detail);
        outStory.innerText = data.story;
        
    } catch(err) {
        outStory.innerHTML = `<span class="error-text">Generation Error: ${err.message}</span>`;
    } finally {
        btnGenerate.disabled = false;
    }
});

// Inference: Similar Words
btnSimilar.addEventListener('click', async () => {
    const word = document.getElementById('inp-word').value.trim();
    if (!word) return;

    outSimilar.innerHTML = '<span class="empty-state">Calculating cosine similarities...</span>';
    btnSimilar.disabled = true;

    try {
        const res = await fetch('/api/similar', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ word: word })
        });
        const data = await res.json();
        
        if (!res.ok) throw new Error(data.detail);

        if (data.results.error) {
            outSimilar.innerHTML = `<span class="error-text">${data.results.error}</span>`;
            return;
        }

        // Render Pills
        outSimilar.innerHTML = '';
        for (const [simWord, score] of Object.entries(data.results)) {
            const pill = document.createElement('div');
            pill.className = 'word-pill';
            pill.innerHTML = `${simWord} <span class="word-score">${score.toFixed(3)}</span>`;
            outSimilar.appendChild(pill);
        }

    } catch(err) {
        outSimilar.innerHTML = `<span class="error-text">Similarity Error: ${err.message}</span>`;
    } finally {
        btnSimilar.disabled = false;
    }
});
