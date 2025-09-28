import React from "react";
import PropTypes from "prop-types";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
} from "recharts";

/**
 * ChartRenderer
 * Accepts a spec object in the shape:
 * { type: 'line'|'bar'|'pie', chart_data: [{x:..., y:...}, ...], title?: string, x?: string, y?: string }
 */
const COLORS = ["#6366f1", "#06b6d4", "#a855f7", "#f59e0b", "#10b981", "#ef4444"];

function simpleAutoDetect(spec) {
  // If spec.type exists, use it. Else detect from data: if many distinct x and y numeric -> line/bar
  const { chart_data } = spec || {};
  if (!chart_data || !Array.isArray(chart_data) || chart_data.length === 0) {
    return { type: "bar", data: [] };
  }
  if (spec.type) return { type: spec.type, data: chart_data };
  // If x values look like dates -> line
  const x0 = chart_data[0]?.x;
  const isDate = x0 && (typeof x0 === "string" && /\d{4}-\d{2}-\d{2}/.test(x0));
  return { type: isDate ? "line" : "bar", data: chart_data };
}

export default function ChartRenderer({ spec }) {
  if (!spec) return null;
  const { type, data } = simpleAutoDetect(spec);
  const title = spec.title || (type === "line" ? "Trend" : "Chart");

  if (!data || data.length === 0) {
    return <div style={{ color: "#94a3b8" }}>No chart data available.</div>;
  }

  if (type === "pie") {
    // Expect data as [{x: label, y: value}, ...]
    return (
      <div style={{ width: "100%", height: 300 }}>
        <h4 style={{ margin: "6px 0 10px" }}>{title}</h4>
        <ResponsiveContainer>
          <PieChart>
            <Pie
              data={data}
              dataKey="y"
              nameKey="x"
              cx="50%"
              cy="50%"
              outerRadius={100}
              label={(entry) => entry.x}
            >
              {data.map((entry, idx) => (
                <Cell key={idx} fill={COLORS[idx % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </div>
    );
  }

  if (type === "line") {
    return (
      <div style={{ width: "100%", height: 320 }}>
        <h4 style={{ margin: "6px 0 10px" }}>{title}</h4>
        <ResponsiveContainer>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="x" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="y" stroke={COLORS[0]} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  }

  // fallback bar
  return (
    <div style={{ width: "100%", height: 320 }}>
      <h4 style={{ margin: "6px 0 10px" }}>{title}</h4>
      <ResponsiveContainer>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="x" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="y" fill={COLORS[0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

ChartRenderer.propTypes = {
  spec: PropTypes.object,
};
