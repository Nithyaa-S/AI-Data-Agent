import React, { useContext } from "react";
import { ThemeContext } from "./ThemeProvider";
import { Sun, Moon } from "lucide-react";

export default function ModeToggle({ className = "" }) {
  const { theme, setTheme } = useContext(ThemeContext);
  const next = theme === "dark" ? "light" : "dark";
  return (
    <button
      type="button"
      onClick={() => setTheme(next)}
      aria-label="Toggle theme"
      className={`inline-flex h-10 w-10 items-center justify-center rounded-lg border border-border hover:bg-muted/40 transition ${className}`}
      title={next === "dark" ? "Switch to Dark" : "Switch to Light"}
    >
      {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
    </button>
  );
}
