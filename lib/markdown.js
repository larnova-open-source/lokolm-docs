// Server-only: render Markdown to an HTML string at build time.
// Using a unified/remark pipeline (instead of a React renderer) keeps this fully
// static and avoids client-side React context during prerender.
import { unified } from "unified";
import remarkParse from "remark-parse";
import remarkGfm from "remark-gfm";
import remarkRehype from "remark-rehype";
import rehypeSlug from "rehype-slug";
import rehypeHighlight from "rehype-highlight";
import rehypeStringify from "rehype-stringify";
import { visit } from "unist-util-visit";

// Rewrite in-repo Markdown links (e.g. "training.md", "./roadmap.md#foo") to app routes.
function rewriteHref(href) {
  if (typeof href !== "string") return href;
  const m = href.match(/(?:^|\/)([\w-]+)\.md(#.*)?$/);
  if (!m) return href;
  const slug = m[1];
  const hash = m[2] || "";
  return (slug === "overview" ? "/" : `/${slug}`) + hash;
}

function rewriteMdLinks() {
  return (tree) => {
    visit(tree, "element", (node) => {
      if (node.tagName === "a" && node.properties) {
        node.properties.href = rewriteHref(node.properties.href);
      }
    });
  };
}

const processor = unified()
  .use(remarkParse)
  .use(remarkGfm)
  .use(remarkRehype)
  .use(rewriteMdLinks)
  .use(rehypeSlug)
  .use(rehypeHighlight)
  .use(rehypeStringify);

export async function renderMarkdown(content) {
  const file = await processor.process(content);
  return String(file);
}
