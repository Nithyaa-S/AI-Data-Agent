import React from "react";
import { twMerge } from "tailwind-merge";

export default function Button({ className = "", variant = "default", size = "md", asChild = false, ...props }) {
  const base = "inline-flex items-center justify-center font-medium transition-colors duration-200 focus:outline-none disabled:opacity-60 disabled:pointer-events-none";
  const sizes = {
    sm: "h-9 px-3 rounded-md text-sm",
    md: "h-10 px-4 rounded-lg text-sm",
    lg: "h-11 px-5 rounded-lg text-base",
  };
  const variants = {
    default: "bg-primary text-primary-foreground hover:opacity-90",
    outline: "border border-border bg-transparent hover:bg-muted/40",
    secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
    ghost: "hover:bg-muted/40",
  };
  const cls = twMerge(base, sizes[size] || sizes.md, variants[variant] || variants.default, className);
  const Comp = asChild ? React.Fragment : "button";
  return <Comp {...props} className={cls} />;
}
