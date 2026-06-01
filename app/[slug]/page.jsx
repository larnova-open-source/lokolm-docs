import { notFound } from "next/navigation";
import { getDoc } from "../../lib/docs.js";
import { renderMarkdown } from "../../lib/markdown.js";
import { DOCS, HOME_SLUG } from "../../lib/docs-meta.js";

// Pre-render every doc except the home doc (served at "/").
export function generateStaticParams() {
  return DOCS.filter((d) => d.slug !== HOME_SLUG).map((d) => ({ slug: d.slug }));
}

export const dynamicParams = false;

export function generateMetadata({ params }) {
  const doc = DOCS.find((d) => d.slug === params.slug);
  return { title: doc ? `lokoLM — ${doc.label}` : "lokoLM — Documentation" };
}

export default async function DocPage({ params }) {
  const doc = getDoc(params.slug);
  if (!doc) notFound();
  const html = await renderMarkdown(doc.content);
  return <div className="md" dangerouslySetInnerHTML={{ __html: html }} />;
}
