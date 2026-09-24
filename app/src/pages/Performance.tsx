import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { AlertTriangle } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, Legend } from 'recharts'

const tierData = [
  { name: '4B Base', latency: 49, agreement: 88.5 },
  { name: '8B int4', latency: 80, agreement: 92.3 },
  { name: '8B BF16', latency: 64, agreement: 98.1 },
]

const crossSourceData = [
  { name: 'TypeSafe claimed', p50: 70, p95: 500, source: 'official', note: 'US West direct, self-reported' },
  { name: 'Reddit (OpenRouter)', p50: 380, p95: 670, source: '3rd party', note: 'n=1500, 3 datasets' },
  { name: 'Open-Jev (direct)', p50: 295, p95: 330, source: '3rd party', note: 'HTTPS, concurrency=1' },
  { name: 'Open-Jev 2B (local H100)', p50: 85, p95: 134, source: '3rd party', note: 'uncached, loopback' },
  { name: 'Decidex 4B (local GPU)', p50: 49, p95: 76, source: 'local', note: 'RTX 5080, this project' },
  { name: 'Decidex 8B + core-8b (local GPU)', p50: 64, p95: 75, source: 'local', note: 'RTX 4090, this project' },
]

const optimizationData = [
  { phase: 'Baseline', noul: 88.5, mae: 0.129 },
  { phase: 'Distill v3', noul: 86.5, mae: 0.098 },
  { phase: 'Active v4', noul: 88.5, mae: 0.113 },
  { phase: 'Balanced v8', noul: 100, mae: 0.061 },
  { phase: 'BF16 v7', noul: 98.1, mae: 0.071 },
]

const fanoutData = [
  { questions: 1, ms: 49 },
  { questions: 5, ms: 52 },
  { questions: 10, ms: 66 },
  { questions: 20, ms: 89 },
  { questions: 30, ms: 112 },
]

const detailedResults = [
  { metric: 'Choice top-1 agreement', value: '23/23 (100%)', ref: '—' },
  { metric: 'Choice distribution JS divergence', value: '0.0025', ref: '—' },
  { metric: 'Noul decision agreement', value: '51/52 (98.1%)', ref: '—' },
  { metric: 'Noul probability MAE', value: '0.071', ref: '—' },
  { metric: 'Score modal-level agreement', value: '10/11 (90.9%)', ref: '—' },
  { metric: 'Score MAE', value: '0.312', ref: '—' },
  { metric: 'Overall agreement', value: '84/86 (97.7%)', ref: '—' },
  { metric: 'Single question (warm p50)', value: '49–64ms', ref: '70–500ms' },
  { metric: '10-question fan-out', value: '66ms', ref: '—' },
  { metric: 'Long-doc fan-out (cached)', value: '193ms', ref: '—' },
  { metric: '32 concurrent requests', value: 'all 200, 0 5xx', ref: '—' },
]

export default function Performance() {
  return (
    <main className="mx-auto max-w-6xl px-4 py-12 md:px-6">
      <h1 className="mb-2 text-4xl font-bold tracking-tight">Performance</h1>
      <p className="mb-10 text-lg leading-relaxed text-muted-foreground">
        Every number on this page is reproducible — the comparison corpus,
        official answers, and every training dataset is committed to the repo.
        Clone it, run the benchmarks, check us. No cherry-picking, no hidden benchmarks, no trust required.
      </p>

      {/* Cross-source latency */}
      <section className="mb-14">
        <h2 className="mb-5 text-2xl font-semibold tracking-tight">Cross-Source Latency Comparison</h2>

        <Card className="mb-6 border-yellow-500/30 bg-yellow-50/50 dark:border-yellow-500/20 dark:bg-yellow-950/10">
          <CardContent className="flex items-start gap-3 py-4">
            <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-yellow-500" />
            <p className="text-sm leading-relaxed text-muted-foreground">
              <strong className="text-foreground">Important:</strong> Decidex runs on your GPU; the official API is a remote service. The
              table below shows both so you can judge the trade-off: your data
              never leaves the machine vs. a network call, zero per-request
              cost vs. metered, no rate limit vs. capped.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardContent>
            <div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="px-3 py-3 md:px-5 text-left font-semibold">Source</th>
                    <th className="px-3 py-3 md:px-5 text-right font-semibold">p50 (ms)</th>
                    <th className="px-3 py-3 md:px-5 text-right font-semibold">p95 (ms)</th>
                    <th className="px-3 py-3 md:px-5 text-left font-semibold">Origin</th>
                    <th className="px-3 py-3 md:px-5 text-left font-semibold">Conditions</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {crossSourceData.map(({ name, p50, p95, source, note }) => (
                    <tr key={name} className={source === 'local' ? 'bg-green-50/50 dark:bg-green-950/10' : ''}>
                      <td className="px-3 py-3 md:px-5 font-medium">{name}</td>
                      <td className="px-3 py-3 md:px-5 text-right font-mono">{p50}</td>
                      <td className="px-3 py-3 md:px-5 text-right font-mono">{p95}</td>
                      <td className="px-3 py-3 md:px-5">
                        <Badge
                          variant={source === 'official' ? 'default' : source === '3rd party' ? 'secondary' : 'outline'}
                          className="whitespace-nowrap text-xs"
                        >
                          {source}
                        </Badge>
                      </td>
                      <td className="px-3 py-3 md:px-5 text-muted-foreground">{note}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* Latency vs Agreement */}
      <section className="mb-14">
        <h2 className="mb-5 text-2xl font-semibold tracking-tight">Latency vs Agreement</h2>
        <Card>
          <CardContent className="p-6">
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={tierData} barCategoryGap="30%">
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(240 5.9% 90%)" />
                <XAxis dataKey="name" tick={{ fontSize: 13 }} />
                <YAxis yAxisId="left" orientation="left" tick={{ fontSize: 12 }} label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fontSize: 12 } }} />
                <YAxis yAxisId="right" orientation="right" domain={[80, 100]} tick={{ fontSize: 12 }} label={{ value: 'Agreement (%)', angle: 90, position: 'insideRight', style: { textAnchor: 'middle', fontSize: 12 } }} />
                <Tooltip
                  contentStyle={{ borderRadius: 8, border: '1px solid hsl(240 5.9% 90%)', fontSize: 13 }}
                  formatter={(value, name) => [name === 'latency' ? `${value}ms` : `${value}%`, name === 'latency' ? 'Latency' : 'Agreement']}
                />
                <Legend wrapperStyle={{ fontSize: 13 }} />
                <Bar yAxisId="left" dataKey="latency" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Latency (ms)" />
                <Bar yAxisId="right" dataKey="agreement" fill="#10b981" radius={[4, 4, 0, 0]} name="Agreement (%)" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </section>

      {/* Optimization journey */}
      <section className="mb-14">
        <h2 className="mb-5 text-2xl font-semibold tracking-tight">Optimization Campaign</h2>
        <p className="mb-5 leading-relaxed text-muted-foreground">
          Five training iterations from frozen baseline to the final BF16 adapter. Noul decision agreement
          and probability MAE both improved through data de-skewing and distillation.
        </p>
        <Card>
          <CardContent className="p-6">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={optimizationData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(240 5.9% 90%)" />
                <XAxis dataKey="phase" tick={{ fontSize: 12 }} />
                <YAxis yAxisId="left" orientation="left" domain={[80, 100]} tick={{ fontSize: 12 }} />
                <YAxis yAxisId="right" orientation="right" domain={[0, 0.15]} tick={{ fontSize: 12 }} />
                <Tooltip
                  contentStyle={{ borderRadius: 8, border: '1px solid hsl(240 5.9% 90%)', fontSize: 13 }}
                />
                <Legend wrapperStyle={{ fontSize: 13 }} />
                <Line yAxisId="left" type="monotone" dataKey="noul" stroke="#10b981" strokeWidth={2.5} name="Noul Agreement (%)" dot={{ r: 5 }} />
                <Line yAxisId="right" type="monotone" dataKey="mae" stroke="#ef4444" strokeWidth={2.5} name="Noul MAE" dot={{ r: 5 }} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </section>

      {/* Fan-out scaling */}
      <section className="mb-14">
        <h2 className="mb-5 text-2xl font-semibold tracking-tight">Fan-out Scaling</h2>
        <p className="mb-5 leading-relaxed text-muted-foreground">
          All questions in one request share the same state via KV prefix reuse — adding questions barely
          increases latency.
        </p>
        <Card>
          <CardContent className="p-6">
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={fanoutData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(240 5.9% 90%)" />
                <XAxis dataKey="questions" tick={{ fontSize: 12 }} label={{ value: 'Questions per Request', position: 'insideBottom', offset: -5, style: { fontSize: 12 } }} />
                <YAxis tick={{ fontSize: 12 }} label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft', style: { textAnchor: 'middle', fontSize: 12 } }} />
                <Tooltip formatter={(v) => [`${v}ms`, 'Latency']} contentStyle={{ borderRadius: 8, fontSize: 13 }} />
                <Line type="monotone" dataKey="ms" stroke="#3b82f6" strokeWidth={3} dot={{ r: 5 }} name="Latency" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </section>

      {/* Detailed results */}
      <section className="mb-14">
        <h2 className="mb-5 text-2xl font-semibold tracking-tight">Detailed Results (core-8b adapter)</h2>
        <Card>
          <CardContent>
            <div>
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/50">
                    <th className="px-3 py-3 md:px-5 text-left font-semibold">Metric</th>
                    <th className="px-3 py-3 md:px-5 text-right font-semibold">Value</th>
                    <th className="px-3 py-3 md:px-5 text-right font-semibold">Official Reference</th>
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {detailedResults.map(({ metric, value, ref }) => (
                    <tr key={metric}>
                      <td className="px-3 py-3 md:px-5">{metric}</td>
                      <td className="px-3 py-3 md:px-5 text-right font-mono">{value}</td>
                      <td className="px-3 py-3 md:px-5 text-right text-muted-foreground">{ref}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </section>

      {/* Optimization techniques */}
      <section>
        <h2 className="mb-5 text-2xl font-semibold tracking-tight">Optimization Techniques</h2>
        <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
          {[
            { title: 'KV Prefix Reuse', desc: 'One forward pass per shared state; every question continues from that cache. 11.4× speedup on long documents.' },
            { title: 'Cross-request LRU Cache', desc: 'Repeat queries over the same state skip prefill entirely (4GB budget). Cold 2.2s → warm 193ms.' },
            { title: 'Batch Chunk Tuning', desc: 'CHUNK=16 measured 20% faster than 8 on fan-out, flat beyond. OOM backoff halves batch automatically.' },
            { title: 'Official-API Distillation', desc: '17,954 samples of the official Jev API\'s actual outputs, 4 rounds + active mining. Noul MAE 0.129 → 0.061.' },
          ].map(({ title, desc }) => (
            <Card key={title}>
              <CardContent className="pt-6">
                <h3 className="mb-2 font-semibold">{title}</h3>
                <p className="text-sm leading-relaxed text-muted-foreground">{desc}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>
    </main>
  )
}
