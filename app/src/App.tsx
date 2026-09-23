import { useState } from 'react'
import { Routes, Route, Link, useLocation } from 'react-router'
import { Github, Zap, Package, BarChart3, Brain, ShieldCheck, Menu, X } from 'lucide-react'
import Home from './pages/Home'
import Compatibility from './pages/Compatibility'
import Performance from './pages/Performance'
import Training from './pages/Training'
import Usage from './pages/Usage'

const nav = [
  { to: '/', label: 'Home', icon: Zap },
  { to: '/usage', label: 'Usage and Models', icon: Package },
  { to: '/performance', label: 'Performance', icon: BarChart3 },
  { to: '/compatibility', label: 'Compatibility', icon: ShieldCheck },
  { to: '/training', label: 'Training', icon: Brain },
]

export default function App() {
  const loc = useLocation()
  const [open, setOpen] = useState(false)

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-50 border-b bg-background/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <Link to="/" className="flex items-center gap-2 font-bold tracking-tight" onClick={() => setOpen(false)}>
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary font-mono text-sm text-primary-foreground">D</span>
            <span className="text-lg">Decidex</span>
          </Link>

          {/* Desktop nav */}
          <nav className="hidden items-center gap-1 md:flex">
            {nav.map(({ to, label, icon: Icon }) => (
              <Link
                key={to}
                to={to}
                className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors ${
                  loc.pathname === to
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                {label}
              </Link>
            ))}
          </nav>

          <div className="flex items-center gap-2">
            <a
              href="https://github.com/fly88oj/decidex"
              target="_blank"
              rel="noopener"
              className="hidden items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground sm:flex"
            >
              <Github className="h-4 w-4" />
              GitHub
            </a>
            <button
              className="rounded-md p-2 hover:bg-muted md:hidden"
              onClick={() => setOpen(!open)}
              aria-label="Toggle menu"
            >
              {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>
        </div>

        {/* Mobile nav drawer */}
        {open && (
          <nav className="border-t bg-background px-4 py-2 md:hidden">
            {nav.map(({ to, label, icon: Icon }) => (
              <Link
                key={to}
                to={to}
                onClick={() => setOpen(false)}
                className={`flex items-center gap-3 rounded-md px-3 py-2.5 text-sm transition-colors ${
                  loc.pathname === to
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                }`}
              >
                <Icon className="h-4 w-4" />
                {label}
              </Link>
            ))}
            <a
              href="https://github.com/fly88oj/decidex"
              target="_blank"
              rel="noopener"
              className="flex items-center gap-3 rounded-md px-3 py-2.5 text-sm text-muted-foreground"
            >
              <Github className="h-4 w-4" />
              GitHub
            </a>
          </nav>
        )}
      </header>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/compatibility" element={<Compatibility />} />
        <Route path="/performance" element={<Performance />} />
        <Route path="/training" element={<Training />} />
        <Route path="/usage" element={<Usage />} />
      </Routes>
      <footer className="border-t px-4 py-8 text-center text-sm text-muted-foreground">
        <p>Decidex — local Jev API replication. MIT License.</p>
        <p className="mt-1">Not affiliated with TypeSafe AI. Jev and TypeSafe are their respective owners' trademarks.</p>
      </footer>
    </div>
  )
}
