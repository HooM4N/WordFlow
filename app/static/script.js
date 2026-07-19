// --- DOM Elements ---
const runSelect = document.getElementById('runSelect');
const loadRunBtn = document.getElementById('loadRunBtn');
const loadStatus = document.getElementById('loadStatus');

const configPanel = document.getElementById('configPanel');
const chartsPanel = document.getElementById('chartsPanel');
const inferencePanel = document.getElementById('inferencePanel');

// Config Elements
const cfgParams = document.getElementById('cfgParams');
const cfgDims = document.getElementById('cfgDims');
const cfgLayers = document.getElementById('cfgLayers');
const cfgDrops = document.getElementById('cfgDrops');
const cfgTie = document.getElementById('cfgTie');
const cfgSeqBatch = document.getElementById('cfgSeqBatch');
const cfgEpochs = document.getElementById('cfgEpochs');
const cfgLr = document.getElementById('cfgLr');
const cfgEs = document.getElementById('cfgEs');
const cfgAmp = document.getElementById('cfgAmp');
const cfgVocab = document.getElementById('cfgVocab');
const cfgLower = document.getElementById('cfgLower');
const cfgSeed = document.getElementById('cfgSeed');
const cfgSplit = document.getElementById('cfgSplit');
const cfgAccel = document.getElementById('cfgAccel');

// Inference Elements
const genPrompt = document.getElementById('genPrompt');
const maxTokens = document.getElementById('maxTokens');
const maxTokensVal = document.getElementById('maxTokensVal');
const temp = document.getElementById('temp');
const tempVal = document.getElementById('tempVal');
const generateBtn = document.getElementById('generateBtn');
const genOutput = document.getElementById('genOutput');

const simWord = document.getElementById('simWord');
const similarBtn = document.getElementById('similarBtn');
const simOutput = document.getElementById('simOutput');

// --- Global State ---
let runsData = [];
let lossChartInstance = null;
let pplChartInstance = null;

// --- Initialize ---
async function init() {
    try {
        const res = await fetch('/api/runs');
        const data = await res.json();
        runsData = data.runs;

        if (runsData.length === 0) {
            runSelect.innerHTML = '<option>No runs found in runs/ directory</option>';
            runSelect.disabled = true;
            return;
        }

        runsData.forEach(run => {
            const opt = document.createElement('option');
            opt.value = run.id;
            opt.textContent = run.id;
            runSelect.appendChild(opt);
        });

        initCharts();

    } catch (e) {
        console.error("Failed to load runs:", e);
    }
}

// --- Formatting Helpers ---
const formatBool = (b) => b ? '<span class="text-emerald-400">Yes</span>' : '<span class="text-rose-400">No</span>';

// --- Charting ---
function initCharts() {
    const lossCtx = document.getElementById('lossChart').getContext('2d');
    const pplCtx = document.getElementById('pplChart').getContext('2d');

    Chart.defaults.color = '#9ca3af'; 
    Chart.defaults.font.family = 'sans-serif';

    lossChartInstance = new Chart(lossCtx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: '#374151' } },
                y: { grid: { color: '#374151' } }
            },
            plugins: { legend: { position: 'top' } }
        }
    });

    pplChartInstance = new Chart(pplCtx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: '#374151' } },
                y: { grid: { color: '#374151' } }
            },
            plugins: { legend: { display: false } }
        }
    });
}

function updateCharts(logs) {
    if (!logs || !logs.train_loss) return;

    const epochs = logs.train_loss.map((_, i) => `Ep ${i + 1}`);

    // Loss Chart
    lossChartInstance.data.labels = epochs;
    lossChartInstance.data.datasets = [
        {
            label: 'Train Loss',
            data: logs.train_loss,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            fill: true,
            tension: 0.4
        }
    ];
    if (logs.val_loss && logs.val_loss.length > 0) {
        lossChartInstance.data.datasets.push({
            label: 'Validation Loss',
            data: logs.val_loss,
            borderColor: '#f59e0b',
            tension: 0.4
        });
    }
    lossChartInstance.update();

    // Perplexity Chart
    if (logs.val_perplexity && logs.val_perplexity.length > 0) {
        pplChartInstance.data.labels = epochs;
        pplChartInstance.data.datasets = [{
            label: 'Validation PPL',
            data: logs.val_perplexity,
            borderColor: '#10b981',
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            fill: true,
            tension: 0.4
        }];
        pplChartInstance.update();
    }
}

// --- Event Listeners ---

// Preview Run Configuration
runSelect.addEventListener('change', (e) => {
    const runId = e.target.value;
    loadRunBtn.disabled = !runId;
    
    if (!runId) {
        configPanel.classList.replace('opacity-100', 'opacity-0');
        configPanel.classList.add('pointer-events-none');
        chartsPanel.classList.replace('opacity-100', 'opacity-0');
        return;
    }

    const selectedRun = runsData.find(r => r.id === runId);
    
    if (selectedRun && selectedRun.config) {
        const c = selectedRun.config;
        
        // Populate Model Params
        cfgDims.textContent = `${c.model.embedding_dim} / ${c.model.hidden_dim}`;
        cfgLayers.textContent = c.model.num_layers;
        cfgDrops.textContent = `${c.model.emb_dropout_p} / ${c.model.rnn_dropout_p} / ${c.model.out_dropout_p}`;
        cfgTie.innerHTML = formatBool(c.model.tie_weights);
        cfgParams.className = "font-mono text-xs bg-gray-900 px-2 py-1 rounded text-gray-400";
        cfgParams.innerHTML = "Load to View"; // Reset parameter count until loaded

        // Populate Training Params
        cfgSeqBatch.textContent = `${c.train.seq_len} / ${c.train.batch_size}`;
        cfgEpochs.textContent = c.train.n_epochs;
        cfgLr.textContent = c.train.lr;
        cfgEs.textContent = c.train.early_stopping_patience;
        cfgAmp.innerHTML = formatBool(c.train.enable_mixed_precision);

        // Populate Data Params
        cfgVocab.textContent = c.data.max_vocab_size.toLocaleString();
        cfgLower.innerHTML = formatBool(c.data.tokenizer_lowercase);
        cfgSeed.textContent = c.seed;
        cfgSplit.textContent = c.data.do_val_split ? c.data.val_split_ratio : "None";
        cfgAccel.innerHTML = formatBool(c.use_accelerator);

        // Show Panels
        configPanel.classList.replace('opacity-0', 'opacity-100');
        configPanel.classList.remove('pointer-events-none');
        
        // Reset load status
        loadStatus.classList.add('hidden');
        inferencePanel.classList.add('opacity-50', 'pointer-events-none');
    }

    if (selectedRun && selectedRun.logs) {
        updateCharts(selectedRun.logs);
        chartsPanel.classList.replace('opacity-0', 'opacity-100');
    }
});

// Update slider labels dynamically
maxTokens.addEventListener('input', e => maxTokensVal.textContent = e.target.value);
temp.addEventListener('input', e => tempVal.textContent = e.target.value);

// Load Model into GPU
loadRunBtn.addEventListener('click', async () => {
    const runId = runSelect.value;
    if (!runId) return;

    loadRunBtn.disabled = true;
    runSelect.disabled = true;
    loadStatus.className = 'mt-3 text-sm text-center text-blue-400';
    loadStatus.innerHTML = '<span class="spinner"></span> Loading into memory...';
    loadStatus.classList.remove('hidden');

    try {
        const res = await fetch('/api/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ run_id: runId })
        });
        
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail);

        // Update Total Parameters now that the model is built in Python
        cfgParams.textContent = data.stats.parameters.toLocaleString();
        cfgParams.className = "font-mono text-sm font-bold text-emerald-400";

        // Enable Inference UI
        inferencePanel.classList.remove('opacity-50', 'pointer-events-none');

        loadStatus.className = 'mt-3 text-sm text-center text-emerald-400 font-semibold';
        loadStatus.innerHTML = '<i class="fa-solid fa-check"></i> Model Ready!';
        
    } catch (e) {
        loadStatus.className = 'mt-3 text-sm text-center text-rose-400';
        loadStatus.textContent = e.message;
    } finally {
        loadRunBtn.disabled = false;
        runSelect.disabled = false;
    }
});

// Generate Text
generateBtn.addEventListener('click', async () => {
    const originalText = generateBtn.innerHTML;
    generateBtn.innerHTML = '<span class="spinner"></span> Generating...';
    generateBtn.disabled = true;
    genOutput.innerHTML = '<span class="animate-pulse">Thinking...</span>';

    try {
        const res = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: genPrompt.value || null,
                max_tokens: parseInt(maxTokens.value),
                temperature: parseFloat(temp.value)
            })
        });
        
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail);
        
        genOutput.innerHTML = data.generated_text.replace(/\n/g, '<br>');
        
    } catch (e) {
        genOutput.innerHTML = `<span class="text-rose-400">${e.message}</span>`;
    } finally {
        generateBtn.innerHTML = originalText;
        generateBtn.disabled = false;
    }
});

// Embedding Similarity Search
similarBtn.addEventListener('click', async () => {
    const word = simWord.value.trim();
    if (!word) return;

    similarBtn.innerHTML = '<span class="spinner"></span>';
    similarBtn.disabled = true;

    try {
        const res = await fetch('/api/similar', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ word: word, top_n: 5 })
        });
        
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail);
        
        const results = data.similar_words;
        
        if (results.error) {
            simOutput.innerHTML = `<p class="text-rose-400 text-sm">${results.error}</p>`;
            return;
        }

        let html = '<ul class="space-y-4">';
        for (const [simWord, score] of Object.entries(results)) {
            const percentage = (score * 100).toFixed(1);
            html += `
                <li>
                    <div class="flex justify-between text-sm mb-1">
                        <span class="font-semibold text-gray-200">${simWord}</span>
                        <span class="text-gray-400">${percentage}%</span>
                    </div>
                    <div class="sim-bar-container">
                        <div class="sim-bar-fill" style="width: ${percentage}%"></div>
                    </div>
                </li>
            `;
        }
        html += '</ul>';
        simOutput.innerHTML = html;
        
    } catch (e) {
        simOutput.innerHTML = `<span class="text-rose-400 text-sm">${e.message}</span>`;
    } finally {
        similarBtn.innerHTML = 'Search';
        similarBtn.disabled = false;
    }
});

// Allow Enter key to trigger similarity search
simWord.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        similarBtn.click();
    }
});

// Run Init
init();
