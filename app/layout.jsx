import "./globals.css";
import "highlight.js/styles/github-dark.css";
import Sidebar from "../components/Sidebar.jsx";

export const metadata = {
  title: "lokoLM — Documentation",
  description:
    "Documentation for lokoLM, a minimal decoder-only Transformer language model for teaching and research.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <div className="layout">
          <Sidebar />
          <main>{children}</main>
        </div>
      </body>
    </html>
  );
}
