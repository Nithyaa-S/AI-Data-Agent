import React from "react";

export default function QueryHistory({ items = [], onSelect }) {
  if (!items.length) {
    return <div className="p-3 text-sm text-muted-foreground">No history yet.</div>;
  }
  return (
    <div className="p-1">
      {items.map((it, i) => (
        <button
          key={`${it.at}-${i}`}
          onClick={() => onSelect?.(it.q)}
          className="w-full text-left px-3 py-2 rounded-md hover:bg-muted/40 transition text-sm"
          title={it.at}
        >
          {it.q}
        </button>
      ))}
    </div>
  );
}
