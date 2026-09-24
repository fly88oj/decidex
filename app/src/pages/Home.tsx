import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { Zap, Shield, Cpu, Download, ArrowRight } from 'lucide-react'
import { Link } from 'react-router'
import { CodeBlock } from '@/components/CodeBlock'

const latencyData = [
  { name: '4B Base', ms: 49, color: '#3b82f6' },
  { name: '8B int4', ms: 80, color: '#8b5cf6' },
  { name: '8B BF16', ms: 64, color: '#10b981' },
  { name: 'Jev (3rd-party)', ms: 295, color: '#f59e0b' },
]

const agreementData = [
  { name: 'Choice top-1', pct: 100, color: '#10b981' },
  { name: 'Noul decisions', pct: 98.1, color: '#3b82f6' },
  { name: 'Score modal', pct: 90.9, color: '#8b5cf6' },
  { name: 'Overall', pct: 97.7, color: '#f59e0b' },
]

export default function Home() {
  return (
    <main className="mx-auto max-w-6xl px-4 py-12 md:px-6">
      {/* Hero */}
      <section className="mb-16 text-center">
        <img src="logos/logo.svg" alt="Decidex logo" className="mx-auto mb-4 h-20 w-20 rounded-2xl" />
        <Badge variant="outline" className="mb-4 text-xs tracking-widest">OPEN SOURCE · LOCAL · API-COMPATIBLE</Badge>
        <h1 className="mb-4 text-5xl font-bold tracking-tight">
          State in. Typed decisions out.
          <br />
          <span className="text-muted-foreground">One forward pass.</span>
        </h1>
        <p className="mx-auto mb-8 max-w-2xl text-lg text-muted-foreground">
          An open reimplementation of the Jev decision model that runs entirely on
          your machine. Your data never leaves your GPU — no API key to buy,
          no rate limit to hit, no cloud to trust. Clone the repo, run one
          command, and you have a decision engine that answers in ~50
          milliseconds.
        </p>
        <div className="flex flex-wrap items-center justify-center gap-3">
          <Link
            to="/usage"
            className="flex items-center gap-2 rounded-lg bg-primary px-6 py-3 text-sm font-medium text-primary-foreground hover:bg-primary/90"
          >
            Get Started <ArrowRight className="h-4 w-4" />
          </Link>
          <Link
            to="/performance"
            className="rounded-lg border px-6 py-3 text-sm font-medium hover:bg-muted"
          >
            View Benchmarks
          </Link>
        </div>
      </section>

      {/* Key metrics */}
      <section className="mb-16 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[
          { icon: Zap, label: 'Latency', value: '49ms', sub: 'on your own GPU — no network round-trip' },
          { icon: Shield, label: 'API Compatible', value: '100%', sub: 'swap one base URL, both official SDKs pass all 8 checks' },
          { icon: Cpu, label: 'Agreement', value: '97.7%', sub: '84/86 vs official Jev, independently measured' },
          { icon: Download, label: 'GGUF', value: '5.0GB', sub: 'fits a gaming laptop, runs offline at zero per-call cost' },
        ].map(({ icon: Icon, label, value, sub }) => (
          <Card key={label}>
            <CardContent className="flex flex-col items-center gap-1 pt-6">
              <Icon className="mb-2 h-6 w-6 text-primary" />
              <div className="text-3xl font-bold">{value}</div>
              <div className="text-sm font-medium">{label}</div>
              <div className="text-xs text-muted-foreground">{sub}</div>
            </CardContent>
          </Card>
        ))}
      </section>

      {/* Charts */}
      <section className="mb-16 grid grid-cols-1 gap-6 lg:grid-cols-2 items-stretch">
        <Card className="flex flex-col">
          <CardHeader>
            <CardTitle>Latency Comparison</CardTitle>
            <CardDescription>
              Single question, warm cache. Local GPU vs Jev API direct (Open-Jev project, third-party
              measured). Network overhead makes these not directly comparable.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex-1">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={latencyData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v) => [`${v}ms`, 'Latency']} />
                <Bar dataKey="ms" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                  {latencyData.map((d, i) => (
                    <Cell key={i} fill={d.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
        <Card className="flex flex-col">
          <CardHeader>
            <CardTitle>Agreement with Official Jev</CardTitle>
            <CardDescription>86-question comparison corpus, core-8b adapter</CardDescription>
          </CardHeader>
          <CardContent className="flex-1">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={agreementData}>
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} />
                <Tooltip formatter={(v) => [`${v}%`, 'Agreement']} />
                <Bar dataKey="pct" radius={[4, 4, 0, 0]} isAnimationActive={false}>
                  {agreementData.map((d, i) => (
                    <Cell key={i} fill={d.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </section>

      {/* Architecture */}
      <section className="mb-16">
        <Card>
          <CardHeader>
            <CardTitle>Architecture</CardTitle>
            <CardDescription>How Decidex works under the hood</CardDescription>
          </CardHeader>
          <CardContent>
            <img
              src="diagrams/architecture.svg"
              alt="Decidex architecture diagram"
              className="mx-auto w-full max-w-3xl"
            />
          </CardContent>
        </Card>
      </section>

      {/* Three primitives */}
      <section className="mb-16 grid grid-cols-1 gap-4 md:grid-cols-3">
        {[
          { title: 'Choice', desc: 'Pick one from up to 255 options. Returns choice, probabilities, and confidence.', example: 'dept → billing (p=0.99)' },
          { title: 'Score', desc: 'Rate on 1–26 ordered levels. Score = Σ i·pᵢ, can land between levels.', example: 'frustration → 1.04 / 2' },
          { title: 'Noul', desc: 'Yes/no as probability [0,1]. The value itself is the belief — no confidence needed.', example: 'urgent → 0.95' },
        ].map(({ title, desc, example }) => (
          <Card key={title}>
            <CardHeader>
              <CardTitle className="font-mono">{title}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="mb-3 text-sm text-muted-foreground">{desc}</p>
              <code className="rounded bg-muted px-2 py-1 text-xs">{example}</code>
            </CardContent>
          </Card>
        ))}
      </section>

      {/* Model downloads */}
      <section className="mb-16">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Download className="h-5 w-5 text-primary" /> Download from HuggingFace
            </CardTitle>
            <CardDescription>
              All adapters and GGUF builds are published and ready to use.
              Base models (Qwen3-4B / 8B) download automatically from Qwen's official page on first run.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {[
                { name: 'decidex-core-8b', desc: 'Flagship adapter', url: 'https://huggingface.co/fly88oj/decidex-core-8b' },
                { name: 'decidex-true-8b', desc: 'Noul-perfect adapter', url: 'https://huggingface.co/fly88oj/decidex-true-8b' },
                { name: 'decidex-gguf', desc: 'GGUF Q4/Q6/Q8 builds', url: 'https://huggingface.co/fly88oj/decidex-gguf' },
              ].map(({ name, desc, url }) => (
                <a key={name} href={url} target="_blank" rel="noopener"
                   className="flex items-center gap-3 rounded-lg border p-4 transition-colors hover:bg-muted">
                  <Download className="h-5 w-5 shrink-0 text-primary" />
                  <div>
                    <div className="font-mono text-sm font-medium">{name}</div>
                    <div className="text-xs text-muted-foreground">{desc}</div>
                  </div>
                </a>
              ))}
            </div>
          </CardContent>
        </Card>
      </section>

      {/* Quick start */}
      <section>
        <Card>
          <CardHeader>
            <CardTitle>Quick Start</CardTitle>
            <CardDescription>Running on your own machine in under 2 minutes — no API key, no account, no cloud</CardDescription>
          </CardHeader>
          <CardContent>
            <CodeBlock>
{`# Install
pip install -e ".[all]"

# Start the service
python -m decidex serve --engine llm --model Qwen/Qwen3-4B --port 8600

# Ask a question
curl -s http://127.0.0.1:8600/v1/systemone \\
  -H "Content-Type: application/json" \\
  -d '{
    "state": "Help! My payouts have been failing.",
    "model": "decidex-latest",
    "questions": {
      "urgent": {"type": "noul", "instructions": "Is this urgent?"}
    }
  }'`}
            </CodeBlock>
          </CardContent>
        </Card>
      </section>
    </main>
  )
}
