import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";
const API_ASK = `${API_BASE}/ask`;
const API_UPLOAD = `${API_BASE}/upload`;
const API_STATUS = `${API_BASE}/status`;

// ─────────────────────────────────────────────────────────────
// MINI MARKDOWN RENDERER
// Handles: **bold**, *italic*, `code`, ## headers, bullet lists
// ─────────────────────────────────────────────────────────────

function renderMarkdown(text) {
  if (!text) return null;

  const lines = text.split("\n");
  const elements = [];
  let i = 0;
  let keyCounter = 0;

  while (i < lines.length) {
    const line = lines[i];

    // Bullet list
    if (/^[\-\*\•]\s+/.test(line)) {
      const listItems = [];
      while (i < lines.length && /^[\-\*\•]\s+/.test(lines[i])) {
        listItems.push(
          <li key={keyCounter++}>{inlineFormat(lines[i].replace(/^[\-\*\•]\s+/, ""))}</li>
        );
        i++;
      }
      elements.push(<ul key={keyCounter++}>{listItems}</ul>);
      continue;
    }

    // Numbered list
    if (/^\d+\.\s+/.test(line)) {
      const listItems = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i])) {
        listItems.push(
          <li key={keyCounter++}>{inlineFormat(lines[i].replace(/^\d+\.\s+/, ""))}</li>
        );
        i++;
      }
      elements.push(<ol key={keyCounter++}>{listItems}</ol>);
      continue;
    }

    // H3
    if (/^###\s+/.test(line)) {
      elements.push(<h4 key={keyCounter++}>{inlineFormat(line.replace(/^###\s+/, ""))}</h4>);
      i++;
      continue;
    }

    // H2
    if (/^##\s+/.test(line)) {
      elements.push(<h3 key={keyCounter++}>{inlineFormat(line.replace(/^##\s+/, ""))}</h3>);
      i++;
      continue;
    }

    // H1
    if (/^#\s+/.test(line)) {
      elements.push(<h3 key={keyCounter++}>{inlineFormat(line.replace(/^#\s+/, ""))}</h3>);
      i++;
      continue;
    }

    // Empty line → paragraph break
    if (line.trim() === "") {
      elements.push(<br key={keyCounter++} />);
      i++;
      continue;
    }

    // Regular paragraph
    elements.push(<p key={keyCounter++}>{inlineFormat(line)}</p>);
    i++;
  }

  return elements;
}

function inlineFormat(text) {
  // Split on bold (**text**), italic (*text*), inline code (`code`),
  // source citations [Source N] and [Visual Source N]
  const parts = [];
  const regex = /(\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`|\[(?:Visual )?Source \d+\])/g;
  let lastIndex = 0;
  let match;
  let k = 0;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    const full = match[0];

    if (full.startsWith("**")) {
      parts.push(<strong key={k++}>{match[2]}</strong>);
    } else if (full.startsWith("*")) {
      parts.push(<em key={k++}>{match[3]}</em>);
    } else if (full.startsWith("`")) {
      parts.push(<code key={k++}>{match[4]}</code>);
    } else if (full.startsWith("[")) {
      parts.push(<span key={k++} className="citation-tag">{full}</span>);
    }

    lastIndex = match.index + full.length;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts.length > 0 ? parts : text;
}



// ─────────────────────────────────────────────────────────────
// APP
// ─────────────────────────────────────────────────────────────

export default function App() {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [statusMsg, setStatusMsg] = useState("Ready");
  const [processingStep, setProcessingStep] = useState(0);
  const [totalSteps] = useState(5);
  const [dragOver, setDragOver] = useState(false);
  const [backendOnline, setBackendOnline] = useState(false);

  const chatEndRef = useRef(null);
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);
  const pollRef = useRef(null);

  // ── Auto-scroll ──────────────────────────────────────────

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // ── Backend health check ─────────────────────────────────

  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(3000) });
        setBackendOnline(res.ok);
      } catch {
        setBackendOnline(false);
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  // ── Status polling (during upload) ───────────────────────

  function startPolling() {
    if (pollRef.current) return;
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch(API_STATUS);
        const data = await res.json();
        setStatusMsg(data.message || "Processing...");
        setProcessingStep(data.step || 0);
        if (data.documents && data.documents.length > 0) {
          setDocuments(data.documents);
        }
        if (!data.processing) {
          stopPolling();
          setUploading(false);
          if (data.documents && data.documents.length > 0) {
            setDocuments(data.documents);
          }
          setStatusMsg(data.message || "Ready");
        }
      } catch {
        // ignore
      }
    }, 1500);
  }

  function stopPolling() {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }

  useEffect(() => () => stopPolling(), []);

  // ── File upload ───────────────────────────────────────────

  const handleFiles = useCallback(async (files) => {
    if (!files || files.length === 0) return;

    const pdfFiles = Array.from(files).filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    if (pdfFiles.length === 0) {
      alert("Please upload PDF files only.");
      return;
    }

    setUploading(true);
    setDocuments([]);
    setStatusMsg("Uploading...");
    setProcessingStep(0);

    const formData = new FormData();
    pdfFiles.forEach((f) => formData.append("files", f));

    try {
      const response = await fetch(API_UPLOAD, { method: "POST", body: formData });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: "Upload failed." }));
        throw new Error(err.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();

      if (data.documents) setDocuments(data.documents);
      setStatusMsg("Processing pipeline started...");

      // Start polling /status for progress
      startPolling();

    } catch (error) {
      console.error("Upload error:", error);
      setUploading(false);
      setStatusMsg("Upload failed");
      setMessages((prev) => [
        ...prev,
        { type: "error", text: `Upload failed: ${error.message}` },
      ]);
    }
  }, []);

  function handleUploadClick(event) {
    handleFiles(event.target.files);
    event.target.value = null;
  }

  // ── Drag & Drop ───────────────────────────────────────────

  function handleDragOver(e) {
    e.preventDefault();
    setDragOver(true);
  }

  function handleDragLeave(e) {
    e.preventDefault();
    setDragOver(false);
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    handleFiles(e.dataTransfer.files);
  }

  // ── Ask question ─────────────────────────────────────────

  async function askQuestion() {
    const trimmed = question.trim();
    if (!trimmed || loading) return;

    setMessages((prev) => [...prev, { type: "user", text: trimmed }]);
    setQuestion("");
    setLoading(true);

    try {
      const response = await fetch(API_ASK, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: "Server error." }));
        throw new Error(err.detail || `HTTP ${response.status}`);
      }

      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        {
          type: "assistant",
          text: data.answer || "No answer returned.",
          sources: data.sources || [],
        },
      ]);
    } catch (error) {
      console.error("Ask error:", error);
      setMessages((prev) => [
        ...prev,
        {
          type: "error",
          text:
            "Could not reach the research agent. Make sure the FastAPI backend is running on port 8000.",
        },
      ]);
    } finally {
      setLoading(false);
      setTimeout(() => textareaRef.current?.focus(), 50);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      askQuestion();
    }
  }


  function clearChat() {
    if (loading) return;
    setMessages([]);
    setQuestion("");
    setTimeout(() => textareaRef.current?.focus(), 50);
  }

  // ── Render ────────────────────────────────────────────────

  const progressPct = totalSteps > 0 ? Math.round((processingStep / totalSteps) * 100) : 0;

  return (
    <div className="app" onDragOver={handleDragOver} onDragLeave={handleDragLeave} onDrop={handleDrop}>

      {/* ──────────────────── DRAG OVERLAY ──────────────────── */}
      {dragOver && (
        <div className="drag-overlay">
          <div className="drag-overlay-inner">
            <span className="drag-icon">📂</span>
            <p>Drop PDFs to upload</p>
          </div>
        </div>
      )}

      {/* ──────────────────── SIDEBAR ────────────────────────── */}
      <aside className="sidebar">

        {/* Brand */}
        <div className="brand">
          <div className="brand-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
              <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
            </svg>
          </div>
          <div>
            <h1>Research Agent</h1>
            <p>Local Multimodal AI</p>
          </div>
        </div>

        {/* Upload zone */}
        <div className="sidebar-section">
          <div className="section-header">
            <span className="section-label">DOCUMENTS</span>
            <label
              className={`upload-btn ${uploading ? "uploading" : ""}`}
              title="Upload PDF files"
            >
              {uploading ? (
                <span className="upload-spinner" />
              ) : (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                  <polyline points="17 8 12 3 7 8"/>
                  <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
              )}
              {uploading ? "Processing…" : "Upload PDF"}
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf"
                style={{ display: "none" }}
                onChange={handleUploadClick}
                disabled={uploading}
              />
            </label>
          </div>

          {/* Progress bar */}
          {uploading && (
            <div className="progress-container">
              <div className="progress-bar" style={{ width: `${progressPct}%` }} />
              <p className="progress-msg">{statusMsg}</p>
            </div>
          )}

          {/* Document list */}
          <div className="document-list">
            {documents.length === 0 ? (
              <div className="doc-empty">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.35">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                  <polyline points="14 2 14 8 20 8"/>
                </svg>
                <p>No documents uploaded.<br/>Drag &amp; drop or click Upload PDF.</p>
              </div>
            ) : (
              documents.map((doc, idx) => (
                <div className="doc-card" key={idx}>
                  <span className="doc-icon">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                      <polyline points="14 2 14 8 20 8"/>
                      <line x1="16" y1="13" x2="8" y2="13"/>
                      <line x1="16" y1="17" x2="8" y2="17"/>
                      <polyline points="10 9 9 9 8 9"/>
                    </svg>
                  </span>
                  <div className="doc-info">
                    <strong title={doc}>{doc.length > 28 ? doc.slice(0, 25) + "…" : doc}</strong>
                    <span>PDF · Ready</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Spacer */}
        <div style={{ flex: 1 }} />

        {/* System status */}
        <div className="system-status">
          <div className={`status-dot ${backendOnline ? "online" : "offline"}`} />
          <div>
            <strong>{backendOnline ? "Backend online" : "Backend offline"}</strong>
            <span>FAISS + BM25 + Qwen2.5-VL</span>
          </div>
        </div>
      </aside>


      {/* ──────────────────── MAIN ───────────────────────────── */}
      <main className="main">

        {/* Top bar */}
        <header className="topbar">
          <div className="topbar-left">
            <h2>Research Assistant</h2>
            <p>
              {documents.length > 0
                ? `${documents.length} document${documents.length > 1 ? "s" : ""} loaded — ask anything`
                : "Upload a PDF document to begin"}
            </p>
          </div>
          <div className="topbar-actions">
            {messages.length > 0 && (
              <button className="clear-button" onClick={clearChat} disabled={loading} title="Clear chat history">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="3 6 5 6 21 6"/>
                  <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
                  <path d="M10 11v6"/><path d="M14 11v6"/>
                  <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
                </svg>
                Clear chat
              </button>
            )}
            <div className="model-badge">
              <span className="model-dot" />
              qwen2.5-vl
            </div>
          </div>
        </header>

        {/* Chat area */}
        <section className="chat">

          {/* Welcome screen */}
          {messages.length === 0 && !loading && (
            <div className="welcome">
              <div className="welcome-glow" />

              {documents.length === 0 ? (
                // No documents yet — show upload prompt
                <>
                  <div className="welcome-icon upload-state">
                    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                      <polyline points="17 8 12 3 7 8"/>
                      <line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                  </div>
                  <h3>Upload your documents to begin</h3>
                  <p>
                    Upload one or more PDF files using the <strong>Upload PDF</strong> button in the sidebar.
                    Once processed, ask any question — I'll search across all your documents and
                    provide accurate, grounded answers with page-level sources.
                  </p>
                  <div className="welcome-features">
                    <div className="feature-chip">📄 Any PDF document</div>
                    <div className="feature-chip">🔍 Hybrid FAISS + BM25 search</div>
                    <div className="feature-chip">🖼️ Tables, figures & charts</div>
                    <div className="feature-chip">📚 Multiple docs at once</div>
                  </div>
                </>
              ) : (
                // Documents loaded — ready to answer
                <>
                  <div className="welcome-icon ready-state">
                    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="20 6 9 17 4 12"/>
                    </svg>
                  </div>
                  <h3>
                    {documents.length === 1
                      ? `"${documents[0]}" is ready`
                      : `${documents.length} documents ready`}
                  </h3>
                  <p>
                    Ask any question about your {documents.length > 1 ? "documents" : "document"}.
                    I'll search the full content — text, tables, figures, and images — and give you
                    a precise, cited answer.
                  </p>
                  {documents.length > 1 && (
                    <div className="loaded-docs">
                      {documents.map((d, i) => (
                        <span key={i} className="loaded-doc-chip">📄 {d}</span>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {/* Messages */}
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.type}`}>

              <div className="message-meta">
                <span className="message-avatar">
                  {msg.type === "user" ? "U" : msg.type === "error" ? "!" : "AI"}
                </span>
                <span className="message-label">
                  {msg.type === "user" ? "You" : msg.type === "error" ? "Error" : "Research Agent"}
                </span>
              </div>

              <div className="message-content">
                {msg.type === "assistant"
                  ? renderMarkdown(msg.text)
                  : <p>{msg.text}</p>
                }
              </div>

              {/* Sources */}
              {msg.sources && msg.sources.length > 0 && (
                <div className="sources">
                  <div className="sources-title">
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                      <polyline points="14 2 14 8 20 8"/>
                    </svg>
                    Sources
                  </div>
                  <div className="sources-list">
                    {msg.sources.slice(0, 6).map((src, si) => (
                      <div className="source-chip" key={si}>
                        {src.source && (
                          <span className="chip-doc" title={src.source}>
                            {src.source.split("/").pop().replace(".pdf", "")}
                          </span>
                        )}
                        {src.page != null && <span>p.{src.page}</span>}
                        {src.chunk_id != null && <span>#{src.chunk_id}</span>}
                        {src.section && <span className="chip-section">{src.section.slice(0, 30)}</span>}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}

          {/* Loading indicator */}
          {loading && (
            <div className="message assistant">
              <div className="message-meta">
                <span className="message-avatar">AI</span>
                <span className="message-label">Research Agent</span>
              </div>
              <div className="message-content loading-content">
                <div className="thinking-dots">
                  <span /><span /><span />
                </div>
                <span className="loading-text">Searching documents and reasoning…</span>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </section>

        {/* Input area */}
        <div className="input-area">
          <div className="input-wrapper">
            <textarea
              ref={textareaRef}
              id="question-input"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                uploading
                  ? "Processing documents, please wait…"
                  : documents.length === 0
                  ? "Upload a PDF first, then ask your question…"
                  : "Ask anything about your documents…"
              }
              rows={1}
              disabled={loading || uploading}
            />
            <button
              id="send-button"
              className="send-button"
              onClick={askQuestion}
              disabled={loading || uploading || !question.trim()}
              title="Send question (Enter)"
            >
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="12" y1="19" x2="12" y2="5"/>
                <polyline points="5 12 12 5 19 12"/>
              </svg>
            </button>
          </div>
          <p className="input-hint">
            Enter to send · Shift+Enter for new line · Supports multiple PDFs simultaneously
          </p>
        </div>

      </main>
    </div>
  );
}