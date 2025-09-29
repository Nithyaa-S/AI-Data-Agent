import React, { useMemo } from "react";

export default function AIInsights({ history = [] }) {
  const last = history[0]?.res;
  const summary = useMemo(() => {
    if (!last) return "Ask a question to see AI insights.";
    const items = [];
    if (last.table?.rows?.length) {
      items.push(`Rows: ${last.table.rows.length}`);
      if (last.table.columns?.length) items.push(`Cols: ${last.table.columns.length}`);
    }
    if (last.chart?.type) items.push(`Chart: ${last.chart.type}`);
    if (last.answer) {
      const a = String(last.answer);
      items.push(`Answer: ${a.length > 140 ? a.slice(0, 140) + "…" : a}`);
    }
    return items.join(" • ") || "No insights available.";
  }, [last]);

  return <div className="text-sm text-muted-foreground">{summary}</div>;
}
