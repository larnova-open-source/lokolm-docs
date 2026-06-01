import { getDoc } from "../lib/docs.js";
import { renderMarkdown } from "../lib/markdown.js";
import { HOME_SLUG } from "../lib/docs-meta.js";

export default async function Home() {
  const doc = getDoc(HOME_SLUG);
  const html = await renderMarkdown(doc.content);
  return <div className="md" dangerouslySetInnerHTML={{ __html: html }} />;
}
