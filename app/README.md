# Decidex website

Marketing/docs site for [Decidex](https://github.com/fly88oj/decidex) — the
local, open reimplementation of the Jev / TypeSafe System One decision API.

Five pages (Home, Usage and Models, Performance, Compatibility, Training)
behind a HashRouter, so the built `dist/` works from any static path.

```bash
npm install
npm run dev       # http://localhost:3000
npm run build     # dist/
```

- Pages live in `src/pages/`; shared UI in `src/components/ui/` (shadcn).
- Numbers on the Performance/Training pages must match `COMPARISON.md` /
  `REPRODUCE.md` in the repo root — treat those files as the source of truth.
