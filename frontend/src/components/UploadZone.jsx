import React, { useCallback, useRef, useState } from "react";
import axios from "axios";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function UploadZone({ onUploaded }) {
  const [dragOver, setDragOver] = useState(false);
  const [file, setFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragOver(false);
    const f = e.dataTransfer?.files?.[0];
    if (f) setFile(f);
  }, []);

  const onUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    setProgress(0);
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await axios.post(`${API}/api/upload`, form, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 120000,
        onUploadProgress: (evt) => {
          if (!evt.total) return;
          const pct = Math.round((evt.loaded / evt.total) * 100);
          setProgress(pct);
        },
      });
      setProgress(100);
      onUploaded?.(res.data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-3">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        className={
          "relative flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-6 text-center transition " +
          (dragOver ? "border-primary/60 bg-muted/30" : "border-border hover:bg-muted/20")
        }
      >
        <div className="text-sm text-muted-foreground">
          Drag & drop your Excel/CSV here, or
        </div>
        <div className="flex items-center gap-2">
          <input
            ref={inputRef}
            type="file"
            accept=".xlsx,.xls,.xlsb,.csv"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="block w-full text-sm file:mr-4 file:rounded-md file:border file:border-border file:bg-transparent file:px-3 file:py-2 file:text-sm file:text-foreground hover:file:bg-muted/40"
          />
          <button
            type="button"
            className="inline-flex h-10 items-center rounded-lg bg-primary px-4 text-sm font-medium text-primary-foreground transition hover:opacity-90 disabled:opacity-60"
            onClick={onUpload}
            disabled={!file || loading}
          >
            {loading ? "Uploading…" : "Upload"}
          </button>
        </div>
        {file && (
          <div className="text-xs text-muted-foreground">Selected: {file.name}</div>
        )}
        {loading && (
          <div className="w-full h-2 rounded bg-muted/40 overflow-hidden">
            <div
              className="h-full bg-primary transition-[width] duration-200"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}
        {error && <div className="text-sm text-destructive">{error}</div>}
      </div>
      <div className="text-xs text-muted-foreground">Accepted: .xlsx, .xls, .xlsb, .csv</div>
    </div>
  );
}
