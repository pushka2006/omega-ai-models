/**
 * OMEGA NEURAL NETWORK ENGINE: Dashboard Frontend Logic
 */

document.addEventListener("DOMContentLoaded", () => {
    initNav();
    loadSystemStatus();
    initBuilder();
    initTraining();
    loadExperiments();
    loadCheckpoints();
});

function initNav() {
    document.querySelectorAll(".nav-item").forEach(item => {
        item.addEventListener("click", () => {
            const view = item.getAttribute("data-view");
            document.querySelectorAll(".nav-item").forEach(el => el.classList.remove("active"));
            item.classList.add("active");

            document.querySelectorAll(".view-container").forEach(c => c.classList.remove("active-view"));
            const target = document.getElementById(`view-${view}`);
            if (target) target.classList.add("active-view");

            if (view === "experiments") loadExperiments();
            if (view === "checkpoints") loadCheckpoints();
            if (view === "status") loadSystemStatus();
        });
    });
}

async function loadSystemStatus() {
    try {
        const res = await fetch("/api/status");
        const d = await res.json();
        document.getElementById("hw-gpu-name").textContent = d.hardware.gpu_name;
        document.getElementById("hw-active-device").textContent = d.hardware.active_device.toUpperCase();
        document.getElementById("stat-experiments-count").textContent = d.recent_experiments_count;
        document.getElementById("status-json").textContent = JSON.stringify(d, null, 2);
    } catch (e) {
        console.error(e);
    }
}

function initBuilder() {
    const btn = document.getElementById("validate-builder-btn");
    const jsonInput = document.getElementById("builder-json");
    const resultBox = document.getElementById("builder-result");

    if (btn && jsonInput) {
        btn.addEventListener("click", async () => {
            try {
                const spec = JSON.parse(jsonInput.value);
                const res = await fetch("/api/builder/validate", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(spec)
                });
                const data = await res.json();
                resultBox.textContent = JSON.stringify(data, null, 2);
            } catch (err) {
                resultBox.textContent = "JSON Syntax Error: " + err.message;
            }
        });
    }
}

function initTraining() {
    const startBtn = document.getElementById("start-train-btn");
    const logBox = document.getElementById("train-log-box");

    if (startBtn) {
        startBtn.addEventListener("click", async () => {
            const task = document.getElementById("train-task").value;
            const epochs = parseInt(document.getElementById("train-epochs").value);
            const lr = parseFloat(document.getElementById("train-lr").value);
            const opt = document.getElementById("train-opt").value;

            logBox.textContent = `Launching verifiable PyTorch training run...\nTask: ${task} | Epochs: ${epochs} | Optimizer: ${opt}\n`;
            startBtn.disabled = true;

            try {
                const res = await fetch("/api/train/start", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        task: task,
                        epochs: epochs,
                        learning_rate: lr,
                        optimizer: opt,
                        hidden_dims: [64, 32, 16]
                    })
                });
                const data = await res.json();
                startBtn.disabled = false;

                let output = `Training Complete!\nExperiment ID: ${data.experiment_id}\n`;
                output += `Best Metric: ${data.results.best_metric} in ${data.results.training_time_seconds}s\n\n`;
                output += "--- Epoch Progression ---\n";
                data.results.history.forEach(h => {
                    const accStr = h.val_acc ? ` | Val Acc: ${h.val_acc}%` : ` | Val Metric: ${h.val_metric}`;
                    output += `Epoch ${h.epoch}: Train Loss = ${h.train_loss} | Val Loss = ${h.val_loss}${accStr} | ${h.samples_per_sec} samp/s\n`;
                });
                logBox.textContent = output;
                loadExperiments();
            } catch (err) {
                startBtn.disabled = false;
                logBox.textContent = "Training Error: " + err.message;
            }
        });
    }
}

async function loadExperiments() {
    const list = document.getElementById("experiments-list");
    if (!list) return;
    try {
        const res = await fetch("/api/experiments");
        const exps = await res.json();
        list.innerHTML = "";
        if (exps.length === 0) {
            list.innerHTML = "<div style='color:var(--text-muted);'>No experiments recorded yet.</div>";
            return;
        }
        exps.forEach(e => {
            const card = document.createElement("div");
            card.className = "glass stat-card";
            card.innerHTML = `
                <div style="display:flex; justify-content:space-between;">
                    <h4 style="color:#fff;">${e.name}</h4>
                    <span style="color:var(--neon-blue); font-size:12px;">Best: ${e.best_val_metric}</span>
                </div>
                <div style="font-size:11px; color:var(--text-muted);">Epochs: ${e.epochs_trained} | Time: ${e.duration_seconds}s</div>
                <pre style="margin-top:8px; font-size:11px;">${JSON.stringify(e.hyperparameters, null, 1)}</pre>
            `;
            list.appendChild(card);
        });
    } catch (e) {
        console.error(e);
    }
}

async function loadCheckpoints() {
    const list = document.getElementById("checkpoints-list");
    if (!list) return;
    try {
        const res = await fetch("/api/checkpoints");
        const ckpts = await res.json();
        list.innerHTML = "";
        if (ckpts.length === 0) {
            list.innerHTML = "<div style='color:var(--text-muted);'>No saved checkpoints found.</div>";
            return;
        }
        ckpts.forEach(c => {
            const card = document.createElement("div");
            card.className = "glass stat-card";
            card.innerHTML = `
                <h4 style="color:var(--neon-blue);">${c.filename}</h4>
                <div style="font-size:12px; color:var(--text-muted);">Size: ${c.size_mb} MB</div>
            `;
            list.appendChild(card);
        });
    } catch (e) {
        console.error(e);
    }
}
