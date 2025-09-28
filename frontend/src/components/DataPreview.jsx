import React from "react";

export default function DataPreview({ dataset }) {
  if (!dataset) return null;
  const tables = dataset.sheets || [];
  const schema = dataset.schema || {};
  const samples = dataset.samples || {};

  return (
    <div>
      <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 8 }}>
        <div className="kpi">
          <div className="pill">Sheets: {tables.length}</div>
          <div className="pill">Tables: {schema.tables ? schema.tables.length : 0}</div>
          <div className="pill">Dataset id: {dataset.dataset_id}</div>
        </div>
      </div>

      <div style={{ marginTop: 8 }}>
        <h4 style={{ margin: "6px 0 10px" }}>Sheets</h4>
        {tables.map((s) => (
          <div key={s} style={{ marginBottom: 8 }}>
            <strong>{s}</strong>
            <div style={{ fontSize: 13, color: "#94a3b8", marginTop: 6 }}>
              Sample rows:
            </div>
            <div className="table-wrap" style={{ marginTop: 8 }}>
              <table className="table">
                <thead>
                  <tr>
                    {(samples[s] && samples[s][0]) ? Object.keys(samples[s][0]).map((c) => <th key={c}>{c}</th>) : <th>No data</th>}
                  </tr>
                </thead>
                <tbody>
                  {(samples[s] || []).slice(0, 6).map((r, idx) => (
                    <tr key={idx}>
                      {(samples[s] && samples[s][0]) ? Object.keys(samples[s][0]).map((c) => <td key={c}>{String(r[c] ?? "")}</td>) : <td />}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
