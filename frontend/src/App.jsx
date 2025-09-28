import React, { useState } from "react";
import FileUpload from "./components/FileUpload";
import Chat from "./components/Chat";
import DataPreview from "./components/DataPreview";

export default function App() {
  const [dataset, setDataset] = useState(null);

  return (
    <div className="app-shell">
      <header className="header">
        <div className="brand">
          <div className="logo" />
          <div>
            <h1 className="title">Cordly AI</h1>
            <p className="subtitle">Conversational analytics for any Excel — upload, ask, discover insights.</p>
          </div>
        </div>
        <div className="toolbar">
          <span className="badge">Interview Build</span>
        </div>
      </header>

      <div className="hero">
        <div className="hero-text">
          <h2 className="hero-title">Turn messy spreadsheets into clear answers</h2>
          <p className="hero-desc">Handles unnamed columns, bad formats, multi-sheet workbooks, and vague questions with AI planning.</p>
        </div>
      </div>

      <div className="grid">
        <div className="stack">
          <div className="card">
            <h3>1) Upload Excel</h3>
            <FileUpload onUploaded={setDataset} />
            <p className="footer-note">Accepted: .xlsx, .xls, .xlsb, .csv</p>
          </div>

          {dataset && (
            <div className="card">
              <h3>Dataset Overview</h3>
              <DataPreview dataset={dataset} />
            </div>
          )}
        </div>

        <div className="stack">
          <div className="card">
            <h3>2) Ask Questions</h3>
            <Chat dataset={dataset} />
          </div>
        </div>
      </div>
    </div>
  );
}
