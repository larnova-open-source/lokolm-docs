# lokoLM — docs

The documentation site for lokoLM, built with **Next.js (App Router)**. Markdown lives in
[content/](content/) and is rendered to static HTML at build time, then deployed to Netlify.

## Develop

```powershell
npm install
npm run dev        # http://localhost:3000
```

## Build (static export)

```powershell
npm run build      # outputs ./out
```

## Editing docs

All content is Markdown under [content/](content/):

- `overview.md` — architecture overview
- `training.md` — GPU / CUDA training guide
- `inference.md` — generating text from a checkpoint
- `roadmap.md` — the v2 roadmap
- `contributing.md` — how to contribute (training data & checkpoints)

To add a page: drop a new `.md` file in `content/` and add an entry to
[lib/docs-meta.js](lib/docs-meta.js). It appears in the sidebar automatically.

## Contributing & license

See [CONTRIBUTING.md](CONTRIBUTING.md) (also published as the
[Contributing](content/contributing.md) docs page). Released under the [MIT License](LICENSE)
© 2026 Mahmud Suberu / Larnova.

Maintained by **Mahmud Suberu** — Founder & CEO of Larnova. Contact:
[LinkedIn](https://www.linkedin.com/in/mahmud-adinoyi-684020235/).

## Deploy to Netlify

Configured via [netlify.toml](netlify.toml) for a static export:

- **Build command:** `npm run build`
- **Publish directory:** `out`

Point Netlify at this repository (base directory = repo root) and it deploys on push.

## ⚠️ Local note for this Windows machine (path casing)

This machine exposes the home folder under several casings (`C:\Users\USER`,
`...\User`, `...\user`, `...\Mahmud` — all the same physical files). Node and webpack
key their module cache on the exact path string, so launching dev/build from a casing that
doesn't match `%USERPROFILE%` makes React load twice and throws
`Invalid hook call` / `Cannot read properties of null (reading 'useReducer'|'useContext')`.

**Fix:** run from the `%USERPROFILE%` casing. Use the provided launchers, which `cd` there
for you:

```powershell
.\dev.cmd        # dev server, correct casing
.\build.cmd      # static export, correct casing
```

Or `cd C:\Users\USER\Desktop\lokoLM\docs` manually before `npm run dev` / `npm run build`.
This affects **local builds only** — Netlify's Linux build is case-sensitive with a single
path and is unaffected. (Moving the project out of `C:\Users\…`, e.g. to `C:\dev\lokoLM`,
also removes the ambiguity permanently.)
