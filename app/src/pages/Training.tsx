import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { CodeBlock } from '@/components/CodeBlock'

const pipeline = [
  {
    step: '1. Generate Data',
    desc: 'Call the official API 1,300 times via OpenRouter (fan-out: ~14 questions per call → 17,954 samples). Four rounds: broad sweep, score-heavy, balanced v2, active mining.',
    cmd: `python benchmarks/distill_generate.py --states 800\npython benchmarks/distill_generate_v2.py --requests 3000\npython benchmarks/distill_active.py --states 500 --model <base>`,
    cost: '< $0.35 total',
  },
  {
    step: '2. Train Adapter',
    desc: 'LoRA r=32 on Qwen3-8B (frozen base, BF16). Soft-label cross-entropy on the letter-logit readout position — the exact same mechanism as inference.',
    cmd: `python benchmarks/distill_train.py \\\n  --model Qwen/Qwen3-8B --dtype auto \\\n  --dataset /tmp/train-core.jsonl \\\n  --epochs 2 --batch-size 4 --grad-accum 2 \\\n  --lora-r 32 --out benchmarks/adapters/decidex-core-8b`,
    cost: '~80 min on a modern GPU (24GB VRAM)',
  },
  {
    step: '3. Evaluate',
    desc: 'Compare adapter outputs against stored official answers (87-question corpus). No API calls needed — official answers are committed in benchmarks/comparison_raw.json.',
    cmd: `python benchmarks/compare_official.py\npython benchmarks/diagnose_flips.py <adapter>`,
    cost: '~2 min',
  },
  {
    step: '4. Package GGUF',
    desc: 'Merge LoRA into base, convert to GGUF F16, then quantize to Q4_K_M / Q6_K / Q8_0. Also build the 0.6B distilled MTP draft.',
    cmd: `# merge\npython -c "from peft import PeftModel; ..."\n# convert + quantize\nllama-quantize f16.gguf Q4_K_M.gguf Q4_K_M`,
    cost: '~10 min',
  },
]

const dataset = {
  total: 17954,
  kinds: [
    { name: 'noul', count: 8986, color: '#3b82f6' },
    { name: 'score', count: 4541, color: '#8b5cf6' },
    { name: 'choice', count: 4427, color: '#10b981' },
  ],
  domains: ['support tickets', 'code review', 'product reviews', 'policy docs', 'meeting notes', 'resumes', 'fact-checking', 'multilingual (zh/ja)'],
}

export default function Training() {
  return (
    <main className="mx-auto max-w-6xl px-4 py-12 md:px-6">
      <h1 className="mb-2 text-4xl font-bold">Training & Distillation</h1>
      <p className="mb-8 text-lg text-muted-foreground">
        The only public model lineage trained toward the official Jev API's actual
        outputs. Everything is open: the data, the training code, the evaluation
        harness — fork it, audit it, rebuild it from scratch on a single GPU.
      </p>

      {/* Pipeline */}
      <section className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Pipeline</h2>
        <div className="space-y-4">
          {pipeline.map(({ step, desc, cmd, cost }) => (
            <Card key={step}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>{step}</CardTitle>
                  <Badge variant="secondary">{cost}</Badge>
                </div>
                <CardDescription>{desc}</CardDescription>
              </CardHeader>
              <CardContent>
                <CodeBlock>{cmd}</CodeBlock>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Dataset composition */}
      <section className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Dataset Composition</h2>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>{dataset.total.toLocaleString()} samples</CardTitle>
              <CardDescription>Across 4 generation rounds + active mining</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {dataset.kinds.map(({ name, count, color }) => (
                  <div key={name} className="flex items-center gap-3">
                    <div className="w-24 text-sm font-medium">{name}</div>
                    <div className="h-6 flex-1 overflow-hidden rounded-md bg-muted">
                      <div
                        className="h-full rounded-md"
                        style={{ width: `${(count / dataset.total) * 100}%`, backgroundColor: color }}
                      />
                    </div>
                    <div className="w-16 text-right text-sm font-mono">{count.toLocaleString()}</div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>De-skewed Coverage</CardTitle>
              <CardDescription>Fixed from the audited v1 skew (was: choice 10%, 4-5 options only, single domain)</CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm">
                {[
                  'Choice arities 2–26 (full letter capacity)',
                  'Score levels 2/3/10',
                  '24% structured states (dict/list with backtick-path questions)',
                  '8 domains',
                  '~12% CJK (Chinese, Japanese, Korean)',
                  '50/25/25 kind balance (was 62/28/10)',
                ].map((item) => (
                  <li key={item} className="flex items-start gap-2">
                    <span className="mt-1 h-1.5 w-1.5 rounded-full bg-primary" />
                    {item}
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Adapter lineage */}
      <section className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Adapter Lineage</h2>
        <Card>
          <CardContent className="p-6">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="pb-2 text-left">Adapter</th>
                  <th className="pb-2 text-left">Training Data</th>
                  <th className="pb-2 text-right">Noul Agreement</th>
                  <th className="pb-2 text-right">Noul MAE</th>
                  <th className="pb-2 text-right">Choice</th>
                  <th className="pb-2 text-right">Score Modal</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {[
                  ['Baseline (frozen)', 'none', '88.5%', '0.129', '100%', '72.7%'],
                  ['core-8b (r1/v7)', 'core set (7,249)', '98.1%', '0.071', '100%', '90.9%'],
                  ['true-8b (r1/v8)', 'balanced (16,220)', '100%', '0.061', '95.7%', '81.8%'],
                ].map(([adapter, data, ...metrics]) => (
                  <tr key={adapter} className={adapter.startsWith('core') ? 'bg-green-500/5' : ''}>
                    <td className="py-2 font-medium">{adapter}</td>
                    <td className="py-2 text-muted-foreground">{data}</td>
                    {metrics.map((m, i) => (
                      <td key={i} className="py-2 text-right font-mono">{m}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-3 text-xs text-muted-foreground">
              core-8b is the default recommendation (best overall balance). true-8b achieves perfect noul
              (52/52) but trades 1 choice + 1 score item.
            </p>
          </CardContent>
        </Card>
      </section>

      {/* Reproducibility */}
      <section>
        <h2 className="mb-4 text-2xl font-semibold">Reproducibility</h2>
        <Card>
          <CardContent className="pt-6">
            <p className="mb-4 text-sm text-muted-foreground">
              Everything in the published results can be regenerated from the repository. All six base
              distillation datasets (26 MB) are committed in-repo. The official answers for evaluation
              (comparison_raw.json) are also committed, so adapter evaluation needs no API access.
            </p>
            <CodeBlock>
{`# Full pipeline reproduction
git clone https://github.com/fly88oj/decidex
cd decidex && pip install -e ".[all,dev]"

# 1. Rebuild training sets from committed base corpora
cat benchmarks/distill_dataset.jsonl benchmarks/distill_dataset_score.jsonl \\
    benchmarks/distill_dataset_active.jsonl \\
    benchmarks/distill_dataset_active.jsonl > /tmp/train-core.jsonl

# 2. Train the core-8b adapter (~80 min on a modern GPU (24GB VRAM))
python benchmarks/distill_train.py \\
  --model Qwen/Qwen3-8B --dtype auto \\
  --dataset /tmp/train-core.jsonl \\
  --epochs 2 --batch-size 4 --grad-accum 2 --lora-r 32 \\
  --out benchmarks/adapters/decidex-core-8b

# 3. Evaluate against official answers
python benchmarks/compare_official.py`}
            </CodeBlock>
          </CardContent>
        </Card>
      </section>
    </main>
  )
}
