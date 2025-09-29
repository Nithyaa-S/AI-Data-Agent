import React, { useState, useRef } from "react";
import axios from "axios";
import { API_CONFIG, getApiUrl } from "../config";

// Create an axios instance with default config
const api = axios.create({
  baseURL: API_CONFIG.BASE_URL,
  timeout: 120000, // Longer timeout for file uploads
  headers: {
    ...API_CONFIG.HEADERS,
    'Content-Type': 'multipart/form-data' // Will be set automatically by axios for FormData
  }
});

export default function FileUpload({ onUploaded }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  const upload = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await api.post(API_CONFIG.ENDPOINTS.UPLOAD, form);
      onUploaded(res.data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  const onDrop = (ev) => {
    ev.preventDefault();
    setError("");
    const f = ev.dataTransfer.files?.[0];
    if (f) setFile(f);
  };

  return (
    <div className="stack">
      <div
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        style={{ borderRadius: 10 }}
      >
        <div className="row" style={{ alignItems: "center" }}>
          <input
            ref={inputRef}
            type="file"
            accept=".xlsx,.xls,.xlsb,.csv"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          <button className="btn" onClick={upload} disabled={!file || loading}>
            {loading ? "Uploading…" : "Upload"}
          </button>
          <button
            className="btn secondary"
            onClick={() => {
              setFile(null);
              setError("");
              inputRef.current && (inputRef.current.value = "");
            }}
          >
            Clear
          </button>
        </div>
      </div>

      {file && <div style={{ color: "#cbd5e1" }}>Selected: {file.name}</div>}
      {error && <div style={{ color: "var(--danger)" }}>{error}</div>}
      <div style={{ color: "#94a3b8", fontSize: 13 }}>Accepted: .xlsx, .xls, .xlsb, .csv</div>
    </div>
  );
}
