import React, { useState } from "react";
import ModeToggle from "./components/theme/ModeToggle";
import UploadZone from "./components/UploadZone";
import DatasetOverview from "./components/DatasetOverview";
import QueryHistory from "./components/QueryHistory";
import AIInsights from "./components/AIInsights";
import Chat from "./components/Chat";

export default function App() {
  const [dataset, setDataset] = useState(null);
  const [history, setHistory] = useState([]); // { q, at: Date, res }

  const onAsk = (q, res) => {
    setHistory((h) => [{ q, at: new Date().toISOString(), res }, ...h].slice(0, 50));
  };

  return (
    <div className="min-h-screen w-full max-w-screen-2xl mx-auto px-4 md:px-6 pb-10">
      {/* Header */}
      <header className="flex items-center justify-between gap-4 py-4">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-lg bg-gradient-to-br from-cyan-400 via-indigo-500 to-purple-500" />
          <div>
            <h1 className="m-0 text-xl font-bold">AI Data Agent</h1>
            <p className="m-0 text-sm text-muted-foreground">Conversational analytics for any Excel — upload, ask, discover insights.</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <ModeToggle />
        </div>
      </header>

      {/* Hero */}
      <section className="mb-4">
        <h2 className="text-2xl md:text-3xl font-extrabold bg-gradient-to-r from-cyan-400 via-indigo-500 to-purple-500 bg-clip-text text-transparent">
          Turn messy spreadsheets into clear answers
        </h2>
        <p className="text-sm md:text-base text-muted-foreground mt-1">Handles unnamed columns, bad formats, multi-sheet workbooks, and vague questions with AI planning.</p>
      </section>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-[420px_minmax(0,1fr)] gap-4">
        {/* Left Panel */}
        <div className="space-y-4">
          <div className="rounded-xl border bg-card text-card-foreground shadow">
            <div className="p-4 border-b"><h3 className="font-semibold">1) Upload Excel</h3></div>
            <div className="p-4">
              <UploadZone onUploaded={setDataset} />
              <p className="mt-2 text-xs text-muted-foreground">Accepted: .xlsx, .xls, .xlsb, .csv</p>
            </div>
          </div>

          {dataset && (
            <div className="rounded-xl border bg-card text-card-foreground shadow">
              <div className="p-4 border-b"><h3 className="font-semibold">Dataset Overview</h3></div>
              <div className="p-4">
                <DatasetOverview dataset={dataset} />
              </div>
            </div>
          )}

          <div className="rounded-xl border bg-card text-card-foreground shadow">
            <div className="p-4 border-b"><h3 className="font-semibold">Query History</h3></div>
            <div className="p-2">
              <QueryHistory items={history} onSelect={(q) => window.dispatchEvent(new CustomEvent('ask-query', { detail: q }))} />
            </div>
          </div>
        </div>

        {/* Right Panel */}
        <div className="space-y-4">
          <div className="rounded-xl border bg-card text-card-foreground shadow">
            <div className="p-4 border-b"><h3 className="font-semibold">2) Ask Questions</h3></div>
            <div className="p-4">
              <Chat dataset={dataset} onAnswered={onAsk} />
            </div>
          </div>

          <div className="rounded-xl border bg-card text-card-foreground shadow">
            <div className="p-4 border-b"><h3 className="font-semibold">AI Insights</h3></div>
            <div className="p-4">
              <AIInsights history={history} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
