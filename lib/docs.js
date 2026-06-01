// Server-only: reads Markdown files from content/ at build time.
import fs from "node:fs";
import path from "node:path";
import { DOCS } from "./docs-meta.js";

const CONTENT_DIR = path.join(process.cwd(), "content");

export function getDoc(slug) {
  const doc = DOCS.find((d) => d.slug === slug);
  if (!doc) return null;
  const content = fs.readFileSync(path.join(CONTENT_DIR, doc.file), "utf8");
  return { ...doc, content };
}
