import React from "react";
import DataPreview from "./DataPreview";

export default function DatasetOverview({ dataset }) {
  if (!dataset) {
    return <div className="text-sm text-muted-foreground">Upload a dataset to see its sheets and samples.</div>;
  }
  return (
    <div className="space-y-2">
      <DataPreview dataset={dataset} />
    </div>
  );
}
