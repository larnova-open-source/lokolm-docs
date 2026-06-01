"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { DOCS } from "../lib/docs-meta.js";

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <nav>
      <div className="brand">
        loko<span>LM</span>
      </div>
      <div className="tag">decoder-only transformer</div>
      <div className="doc-switch">
        {DOCS.map((doc) => {
          const active =
            doc.href === "/"
              ? pathname === "/"
              : pathname.startsWith(doc.href);
          return (
            <Link
              key={doc.slug}
              href={doc.href}
              className={active ? "active" : ""}
            >
              {doc.label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
