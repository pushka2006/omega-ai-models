/**
 * OMEGA MODEL 2: Single-Page Application Controller
 * Handles real-time chat, 10-view navigation, memory CRUD, knowledge ingestion,
 * sandboxed execution, tool triggering, and system status telemetry.
 */

let activeView = "dashboard";
let currentConvId = null;
let activeProvider = "omega_offline";
let webSearchEnabled = false;

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initChat();
    initCodingAssistant();
    initSearch();
    initKnowledge();
    initMemory();
    initProductivity();
    refreshSystemStatus();
    loadModels();
});

// 1. Navigation Controller
function initNavigation() {
    const navItems = document.querySelectorAll(".nav-item");
    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const targetView = item.getAttribute("data-view");
            switchView(targetView);
        });
    });
}

function switchView(viewName) {
    activeView = viewName;
    document.querySelectorAll(".nav-item").forEach(el => {
        el.classList.toggle("active", el.getAttribute("data-view") === viewName);
    });

    document.querySelectorAll(".view-container").forEach(el => {
        el.classList.toggle("active-view", el.id === `view-${viewName}`);
    });

    // Lazy data refresh per view
    if (viewName === "dashboard" || viewName === "system") refreshSystemStatus();
    if (viewName === "models") loadModels();
    if (viewName === "memory") loadMemories();
    if (viewName === "knowledge") loadDocuments();
    if (viewName === "productivity") loadProductivity();
}

// 2. Chat Controller
function initChat() {
    const sendBtn = document.getElementById("chat-send-btn");
    const input = document.getElementById("chat-input");
    const searchToggle = document.getElementById("web-search-toggle");
    const providerSelect = document.getElementById("provider-select");

    if (searchToggle) {
        searchToggle.addEventListener("change", (e) => {
            webSearchEnabled = e.target.checked;
        });
    }

    if (providerSelect) {
        providerSelect.addEventListener("change", (e) => {
            activeProvider = e.target.value;
        });
    }

    sendBtn.addEventListener("click", () => sendChatMessage());
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });
}

async function sendChatMessage() {
    const input = document.getElementById("chat-input");
    const query = input.value.trim();
    if (!query) return;

    input.value = "";
    appendMessage("user", query);

    const loadingId = appendMessage("assistant", "Thinking & synthesizing...", true);

    try {
        const resp = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: query,
                conversation_id: currentConvId,
                provider: activeProvider,
                use_web_search: webSearchEnabled,
                use_knowledge_base: true,
            })
        });

        const data = await resp.json();
        currentConvId = data.conversation_id;

        // Replace loading message with full structured response
        const msgElem = document.getElementById(loadingId);
        if (msgElem) {
            msgElem.classList.remove("loading");
            msgElem.innerHTML = renderMarkdown(data.response);
        }
    } catch (err) {
        const msgElem = document.getElementById(loadingId);
        if (msgElem) {
            msgElem.innerHTML = `<span style="color:#ef4444;">Error communicating with OMEGA backend: ${err.message}</span>`;
        }
    }
}

function appendMessage(role, text, isLoading = false) {
    const box = document.getElementById("messages-box");
    const msgId = "msg-" + Date.now();
    const div = document.createElement("div");
    div.id = msgId;
    div.className = `message-bubble ${role === "user" ? "user-msg" : "bot-msg"}`;
    if (isLoading) div.classList.add("loading");
    div.innerHTML = role === "user" ? escapeHtml(text) : renderMarkdown(text);
    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
    return msgId;
}

// 3. Coding Assistant Controller
function initCodingAssistant() {
    const runBtn = document.getElementById("code-run-btn");
    const codeArea = document.getElementById("code-input");
    const outputArea = document.getElementById("code-output");

    if (runBtn && codeArea) {
        runBtn.addEventListener("click", async () => {
            const code = codeArea.value.trim();
            if (!code) return;

            outputArea.textContent = "Executing inside secure sandbox...";
            try {
                const res = await fetch("/api/coding/execute", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ code: code, language: "python" })
                });
                const data = await res.json();
                if (data.success) {
                    outputArea.textContent = `[Completed in ${data.execution_time_ms}ms]\n\n${data.output}`;
                } else {
                    outputArea.textContent = `[Error (${data.execution_time_ms}ms)]\n\n${data.error}`;
                }
            } catch (err) {
                outputArea.textContent = `Execution failed: ${err.message}`;
            }
        });
    }
}

// 4. Web Search Controller
function initSearch() {
    const btn = document.getElementById("search-btn");
    const input = document.getElementById("search-query");
    const resultsContainer = document.getElementById("search-results-list");

    if (btn && input) {
        btn.addEventListener("click", async () => {
            const q = input.value.trim();
            if (!q) return;

            resultsContainer.innerHTML = "<div style='color:var(--neon-blue);'>Searching the web via DuckDuckGo...</div>";
            try {
                const resp = await fetch("/api/search", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query: q, max_results: 5 })
                });
                const data = await resp.json();
                resultsContainer.innerHTML = "";
                if (!data.found || data.citations.length === 0) {
                    resultsContainer.innerHTML = "<div>No web search results found.</div>";
                    return;
                }
                data.raw_results.forEach(item => {
                    const card = document.createElement("div");
                    card.className = "glass stat-card";
                    card.style.marginBottom = "12px";
                    card.innerHTML = `
                        <h4 style="color:var(--neon-blue); font-size:15px;"><a href="${item.url}" target="_blank" style="color:inherit; text-decoration:none;">${item.title}</a></h4>
                        <span style="font-size:11px; color:var(--text-muted);">${item.source}</span>
                        <p style="font-size:13px; line-height:1.5; margin-top:6px;">${item.snippet}</p>
                    `;
                    resultsContainer.appendChild(card);
                });
            } catch (e) {
                resultsContainer.innerHTML = `<div style="color:#ef4444;">Search request error: ${e.message}</div>`;
            }
        });
    }
}

// 5. Knowledge Hub Controller
function initKnowledge() {
    const ingestBtn = document.getElementById("ingest-btn");
    const filenameInput = document.getElementById("doc-filename");
    const contentInput = document.getElementById("doc-content");

    if (ingestBtn) {
        ingestBtn.addEventListener("click", async () => {
            const fn = filenameInput.value.trim();
            const ct = contentInput.value.trim();
            if (!fn || !ct) {
                alert("Please provide both filename and content.");
                return;
            }

            try {
                await fetch("/api/knowledge/ingest-text", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ filename: fn, content: ct })
                });
                filenameInput.value = "";
                contentInput.value = "";
                loadDocuments();
            } catch (e) {
                alert("Ingestion error: " + e.message);
            }
        });
    }
}

async function loadDocuments() {
    const list = document.getElementById("knowledge-docs-list");
    if (!list) return;

    try {
        const resp = await fetch("/api/knowledge/documents");
        const docs = await resp.json();
        list.innerHTML = "";
        if (docs.length === 0) {
            list.innerHTML = "<div style='color:var(--text-muted);'>No documents ingested yet.</div>";
            return;
        }
        docs.forEach(d => {
            const card = document.createElement("div");
            card.className = "glass stat-card";
            card.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4 style="color:#fff;">${d.filename}</h4>
                    <span style="font-size:11px; color:var(--neon-blue);">${d.chunk_count} Chunks</span>
                </div>
                <p style="font-size:12px; color:var(--text-muted);">${d.summary || ""}</p>
                <button class="neon-btn" style="padding:4px 10px; font-size:11px; align-self:flex-start; margin-top:8px;" onclick="deleteDocument('${d.id}')">Delete</button>
            `;
            list.appendChild(card);
        });
    } catch (e) {
        console.error("Error loading documents", e);
    }
}

async function deleteDocument(docId) {
    if (confirm("Delete document and its semantic chunk vectors?")) {
        await fetch(`/api/knowledge/documents/${docId}`, { method: "DELETE" });
        loadDocuments();
    }
}

// 6. Memory Controller
function initMemory() {
    const addBtn = document.getElementById("add-memory-btn");
    if (addBtn) {
        addBtn.addEventListener("click", async () => {
            const cat = document.getElementById("mem-cat").value;
            const type = document.getElementById("mem-type").value;
            const content = document.getElementById("mem-content").value.trim();
            if (!content) return;

            await fetch("/api/memory", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ category: cat, memory_type: type, content: content, importance_score: 0.85 })
            });
            document.getElementById("mem-content").value = "";
            loadMemories();
        });
    }
}

async function loadMemories() {
    const container = document.getElementById("memory-items-list");
    if (!container) return;

    try {
        const res = await fetch("/api/memory");
        const mems = await res.json();
        container.innerHTML = "";
        if (mems.length === 0) {
            container.innerHTML = "<div style='color:var(--text-muted);'>No memories stored.</div>";
            return;
        }
        mems.forEach(m => {
            const card = document.createElement("div");
            card.className = "glass stat-card";
            card.innerHTML = `
                <div style="display:flex; justify-content:space-between;">
                    <span style="font-size:11px; color:var(--neon-violet); text-transform:uppercase;">${m.memory_type} | ${m.category}</span>
                    <span style="font-size:11px; color:var(--neon-blue);">Score: ${m.importance_score}</span>
                </div>
                <div style="font-size:13px; line-height:1.5;">${escapeHtml(m.content)}</div>
                <button class="neon-btn" style="padding:4px 8px; font-size:10px; align-self:flex-start; margin-top:6px;" onclick="deleteMemory('${m.id}')">Delete</button>
            `;
            container.appendChild(card);
        });
    } catch (e) {
        console.error(e);
    }
}

async function deleteMemory(id) {
    await fetch(`/api/memory/${id}`, { method: "DELETE" });
    loadMemories();
}

// 7. Productivity Controller
function initProductivity() {
    const addNoteBtn = document.getElementById("add-note-btn");
    const addTaskBtn = document.getElementById("add-task-btn");

    if (addNoteBtn) {
        addNoteBtn.addEventListener("click", async () => {
            const title = document.getElementById("note-title").value.trim();
            const content = document.getElementById("note-content").value.trim();
            if (!title) return;
            await fetch("/api/productivity/notes", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ title, content })
            });
            document.getElementById("note-title").value = "";
            document.getElementById("note-content").value = "";
            loadProductivity();
        });
    }

    if (addTaskBtn) {
        addTaskBtn.addEventListener("click", async () => {
            const title = document.getElementById("task-title").value.trim();
            if (!title) return;
            await fetch("/api/productivity/tasks", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ title })
            });
            document.getElementById("task-title").value = "";
            loadProductivity();
        });
    }
}

async function loadProductivity() {
    // Load notes & tasks
    const notesBox = document.getElementById("notes-list");
    const tasksBox = document.getElementById("tasks-list");

    try {
        const nRes = await fetch("/api/productivity/notes");
        const notes = await nRes.json();
        if (notesBox) {
            notesBox.innerHTML = "";
            notes.forEach(n => {
                const el = document.createElement("div");
                el.className = "glass stat-card";
                el.innerHTML = `<h4>${escapeHtml(n.title)}</h4><p style="font-size:12px; color:var(--text-muted);">${escapeHtml(n.content)}</p>`;
                notesBox.appendChild(el);
            });
        }

        const tRes = await fetch("/api/productivity/tasks");
        const tasks = await tRes.json();
        if (tasksBox) {
            tasksBox.innerHTML = "";
            tasks.forEach(t => {
                const el = document.createElement("div");
                el.className = "glass stat-card";
                el.innerHTML = `<div style="display:flex; justify-content:space-between;"><span>${escapeHtml(t.title)}</span><span style="color:var(--neon-cyan);">${t.status}</span></div>`;
                tasksBox.appendChild(el);
            });
        }
    } catch (e) {
        console.error(e);
    }
}

// 8. Models Controller
async function loadModels() {
    const container = document.getElementById("models-grid");
    const provSelect = document.getElementById("provider-select");
    if (!container) return;

    try {
        const res = await fetch("/api/models");
        const data = await res.json();
        container.innerHTML = "";
        if (provSelect) provSelect.innerHTML = "";

        data.models.forEach(m => {
            const card = document.createElement("div");
            card.className = "glass stat-card";
            const statusColor = m.is_available ? "#10b981" : "#ef4444";
            card.innerHTML = `
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3 style="color:#fff;">${m.name}</h3>
                    <span style="color:${statusColor}; font-size:11px; font-weight:700;">${m.is_available ? "ONLINE" : "OFFLINE"}</span>
                </div>
                <div style="font-size:12px; color:var(--neon-blue); font-family:var(--font-mono);">${m.id} | ${m.type.toUpperCase()}</div>
                <p style="font-size:12px; color:var(--text-muted); margin-top:6px;">${m.description}</p>
            `;
            container.appendChild(card);

            if (provSelect && m.is_available) {
                const opt = document.createElement("option");
                opt.value = m.id;
                opt.textContent = m.name;
                provSelect.appendChild(opt);
            }
        });
    } catch (e) {
        console.error(e);
    }
}

// 9. System Diagnostics
async function refreshSystemStatus() {
    try {
        const res = await fetch("/api/system/status");
        const data = await res.json();

        // Update dashboard metrics
        setElementText("stat-mem-count", data.database_telemetry.memories_count);
        setElementText("stat-doc-count", data.database_telemetry.documents_count);
        setElementText("stat-chunk-count", data.database_telemetry.chunks_count);
        setElementText("stat-conv-count", data.database_telemetry.conversations_count);
        setElementText("stat-tool-count", data.database_telemetry.tool_executions_count);

        setElementText("m1-status", data.model_1_sovereign.status);
        setElementText("device-status", data.environment.device);
    } catch (e) {
        console.error("Failed to load telemetry", e);
    }
}

function setElementText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

// Formatting Helpers
function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderMarkdown(md) {
    if (!md) return "";
    let html = escapeHtml(md);
    // Code blocks
    html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code style="color:var(--neon-cyan); background:rgba(255,255,255,0.06); padding:2px 4px; border-radius:4px;">$1</code>');
    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    // Italics
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    // Headers
    html = html.replace(/### (.*?)\n/g, '<h3 style="color:var(--neon-blue); margin:10px 0 6px 0;">$1</h3>');
    html = html.replace(/## (.*?)\n/g, '<h2 style="color:#ffffff; margin:14px 0 8px 0;">$1</h2>');
    // Newlines
    html = html.replace(/\n/g, '<br/>');
    return html;
}

