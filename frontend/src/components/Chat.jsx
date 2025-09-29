import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import ChartRenderer from "./ChartRenderer";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function Chat({ dataset, onAnswered }) {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]); // {role, content, data?, error?}
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);
  const onAnsweredRef = useRef(onAnswered);

  // keep ref updated when prop changes
  useEffect(() => {
    onAnsweredRef.current = onAnswered;
  }, [onAnswered]);


  // Listen to history selection to prefill input
  useEffect(() => {
    const handler = (e) => {
      const q = e?.detail;
      if (typeof q === "string" && q.trim()) setQuestion(q);
    };
    window.addEventListener("ask-query", handler);
    return () => window.removeEventListener("ask-query", handler);
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const ask = async () => {
    if (!question.trim() || !dataset?.dataset_id) return;
    setLoading(true);
    const q = question.trim();
    setQuestion("");
    setMessages((m) => [...m, { role: "user", content: q }]);
    try {
      const res = await axios.post(`${API}/api/ask`, {
        dataset_id: dataset.dataset_id,
        question: q,
      });
      setMessages((m) => [
        ...m,
        { role: "assistant", content: res.data.answer, data: res.data },
      ]);
    } catch (err) {
      const msg = err.response?.data?.detail || err.response?.data?.answer || err.message || "Unknown error";
      setMessages((m) => [...m, { role: "assistant", content: msg, error: true }]);
    } finally {
      setLoading(false);
    }
  };

  const downloadCSV = (table) => {
    if (!table || !table.columns || !table.rows) return;
    const csv = table.columns.join(",") + "\n" + table.rows.map((r) => table.columns.map((c) => JSON.stringify(r[c] ?? "")).join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "export.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="stack">
      <div className="row">
        <input
          className="input"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="e.g., Top 5 products by revenue in Q2 2024"
          onKeyDown={(e) => e.key === "Enter" && ask()}
        />
        <button className="btn" onClick={ask} disabled={loading || !dataset?.dataset_id}>
          {loading ? "Thinking…" : "Ask"}
        </button>
      </div>

      {!dataset?.dataset_id && <div style={{ color: "#94a3b8" }}>Upload a dataset first to ask questions.</div>}

      <div className="chat" style={{ marginTop: 12 }}>
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role} ${m.error ? "error" : ""}`}>
            <div className="role">{m.role}</div>
            <div style={{ whiteSpace: "pre-wrap" }}>{m.content}</div>

            {m.role === "assistant" && m.data && (
              <>
                {m.data.error && <div style={{ color: "var(--danger)", marginTop: 8 }}>{m.data.error}</div>}

                {m.data.table && m.data.table.rows && m.data.table.rows.length > 0 && (
                  <div style={{ marginTop: 10 }}>
                    <div className="table-wrap">
                      <table className="table">
                        <thead>
                          <tr>
                            {m.data.table.columns.map((c) => (
                              <th key={c}>{c}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {m.data.table.rows.slice(0, 50).map((r, idx) => (
                            <tr key={idx}>
                              {m.data.table.columns.map((c) => (
                                <td key={c}>{String(r[c] ?? "")}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div style={{ marginTop: 8 }}>
                      <button className="btn secondary" onClick={() => downloadCSV(m.data.table)}>
                        ⬇ Download CSV
                      </button>
                    </div>
                  </div>
                )}

                {m.data.chart && (
                  <div style={{ marginTop: 12 }}>
                    <ChartRenderer spec={m.data.chart} />
                  </div>
                )}

                {m.data.sql && (
                  <pre style={{ background: "rgba(255,255,255,0.03)", padding: 10, borderRadius: 10, marginTop: 12, fontFamily: "monospace" }}>
                    {m.data.sql}
                  </pre>
                )}
              </>
            )}
          </div>
        ))}

        {loading && (
          <div className="msg assistant">
            <div className="role">assistant</div>
            <div style={{ opacity: 0.8 }}>Thinking…</div>
          </div>
        )}

        <div ref={chatEndRef} />
      </div>
    </div>
  );
}
