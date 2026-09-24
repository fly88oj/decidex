import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { CodeBlock } from '@/components/CodeBlock'

const builds = [
  { name: 'Q4_K_M', size: '5.0 GB', best: '16GB GPUs / 16GB+ RAM CPU', speed: '~80ms' },
  { name: 'Q6_K', size: '6.7 GB', best: 'quality-first', speed: '~75ms' },
  { name: 'Q8_0', size: '8.7 GB', best: 'closest to BF16 behavior', speed: '~70ms' },
]

const platforms = [
  {
    name: 'llama-server',
    install: `llama-server -m gguf/decidex-core-8b-r1-Q4_K_M.gguf -c 8192 --port 8080`,
    usage: `POST /completion with n_probs for probability distributions`,
    features: ['Full probability readout via n_probs', 'MTP draft support', 'Best overall experience'],
  },
  {
    name: 'Ollama',
    install: `# Modelfile\nFROM ./decidex-core-8b-r1-Q4_K_M.gguf\n\nollama create decidex -f Modelfile`,
    usage: `POST /api/generate with raw:true, temperature:0, num_predict:2`,
    features: ['Simple setup', 'Argmax letter only (no logprobs)', 'Good for quick testing'],
  },
  {
    name: 'LM Studio',
    install: `Import GGUF file via GUI or place in model directory`,
    usage: `POST /v1/completions (OpenAI-compatible, not chat)`,
    features: ['GUI-friendly', 'OpenAI API compatibility', 'Supports top_logprobs'],
  },
]

export default function Usage() {
  return (
    <main className="mx-auto max-w-6xl px-4 py-12 md:px-6">
      <h1 className="mb-2 text-4xl font-bold">Usage and Models</h1>
      <p className="mb-8 text-lg text-muted-foreground">
        Everything runs where you are — a local GPU, a laptop with Ollama, or
        LM Studio on your desktop. No API keys, no rate limits, no data leaving
        your machine. Here's how to use the models, and which variants to pick.
      </p>

      {/* ===== USAGE ===== */}

      {/* Decision prompt template */}
      <section className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">The Decision Prompt</h2>
        <Card>
          <CardHeader>
            <CardDescription>
              Use this exact template with <strong>temperature=0</strong> and <strong>max_tokens=1–4</strong>.
              Do NOT use a chat template — the model is a thinking hybrid and chat templates break the readout.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <CodeBlock>
{`You are a precise decision engine. Evaluate the STATE against the
QUESTION and pick the single best option.

STATE:
{your state here}

QUESTION:
{your question here}

OPTIONS:
A. {option A description}
B. {option B description}
C. {option C description}

Answer with the letter of the single best option.
Answer:`}
            </CodeBlock>
          </CardContent>
        </Card>
      </section>

      {/* Platform guides */}
      <section className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Platform Guides</h2>
        <div className="space-y-4">
          {platforms.map(({ name, install, usage, features }) => (
            <Card key={name}>
              <CardHeader>
                <CardTitle>{name}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <Badge variant="secondary" className="mb-1">Setup</Badge>
                  <code className="rounded bg-muted block p-3 text-xs">{install}</code>
                </div>
                <div>
                  <Badge variant="secondary" className="mb-1">Usage</Badge>
                  <p className="text-sm text-muted-foreground">{usage}</p>
                </div>
                <div className="flex flex-wrap gap-2">
                  {features.map((f) => (
                    <Badge key={f} variant="outline" className="text-xs">{f}</Badge>
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Probability readout */}
      <section className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Reading Probabilities (llama-server)</h2>
        <Card>
          <CardContent className="pt-6">
            <p className="mb-4 text-sm text-muted-foreground">
              The full probability distribution over option letters comes from llama-server's
              <code className="mx-1 rounded bg-muted px-1">n_probs</code> parameter. A ready-made client is at
              <code className="mx-1 rounded bg-muted px-1">examples/gguf_decision_client.py</code>.
            </p>
            <CodeBlock>
{`curl -s http://127.0.0.1:8080/completion \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt": "<decision prompt from above>",
    "n_predict": 1,
    "temperature": 0,
    "n_probs": 20
  }'

# → completion_probabilities[0].top_logprobs
#   has p(A), p(B), p(C) at the answer position

# Verified live: official example returns
#   noul 0.9503 (official: 0.95)
#   score 1.0404 (official: 1.05)`}
            </CodeBlock>
          </CardContent>
        </Card>
      </section>

      {/* MTP / Speculative decoding */}
      <section className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">MTP / Speculative Decoding</h2>
        <Card>
          <CardHeader>
            <CardDescription>
              Qwen3-8B has no checkpoint NextN weights, so MTP uses a draft model — the same mechanism
              an MTP head uses internally. The distilled draft matches the target's letter distribution,
              giving high acceptance rates.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <CodeBlock>
{`llama-server -m gguf/decidex-core-8b-r1-Q4_K_M.gguf \\
  --model-draft gguf/decidex-draft-0.6b-Q8_0.gguf \\
  --spec-draft-n-max 4 -ngl 99`}
            </CodeBlock>
            <p className="mt-3 text-sm text-muted-foreground">
              Note: speculative decoding accelerates generation. Decision readout generates zero tokens,
              so MTP benefits llama-server chat/completions use of the merged model, not the decision
              API itself.
            </p>
          </CardContent>
        </Card>
      </section>

      {/* ===== MODELS ===== */}

      {/* Available builds */}
      <section className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Available GGUF Builds</h2>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          {builds.map(({ name, size, best, speed }) => (
            <Card key={name}>
              <CardContent className="flex flex-col items-center gap-2 pt-6">
                <div className="font-mono text-2xl font-bold">{name}</div>
                <div className="text-lg text-primary">{size}</div>
                <div className="text-sm text-muted-foreground">{best}</div>
                <Badge variant="secondary">{speed}</Badge>
              </CardContent>
            </Card>
          ))}
        </div>
        <p className="mt-3 text-center text-sm text-muted-foreground">
          All builds carry the <code className="rounded bg-muted px-1">r1</code> lineage marker
          (core-8b adapter, first release). MTP draft companion:{' '}
          <code className="rounded bg-muted px-1">decidex-draft-0.6b-Q8_0.gguf</code> (610 MB)
        </p>
      </section>

      {/* Model downloads */}
      <section>
        <h2 className="mb-4 text-2xl font-semibold">Model Downloads (HuggingFace)</h2>
        <p className="mb-4 text-sm text-muted-foreground">
          All adapters and GGUF builds are published on HuggingFace. Download with the CLI or
          <code className="mx-1 rounded bg-muted px-1">scripts/fetch_models.py</code> in the repo.
        </p>
        <p className="mb-4 text-sm text-muted-foreground">
          <strong className="text-foreground">Base models</strong> are downloaded separately from
          Qwen's official HuggingFace page — they are not our artifacts.
          <br />
          4B tier: <code className="rounded bg-muted px-1">Qwen/Qwen3-4B</code> (safetensors,
          auto-downloaded by <code className="rounded bg-muted px-1">transformers</code> on first run).
          <br />
          8B tier: <code className="rounded bg-muted px-1">Qwen/Qwen3-8B</code> (safetensors,
          auto-downloaded by <code className="rounded bg-muted px-1">transformers</code> on first run).
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="px-3 py-3 md:px-5 text-left font-semibold">Model</th>
                <th className="px-3 py-3 md:px-5 text-left font-semibold">Type</th>
                <th className="px-3 py-3 md:px-5 text-left font-semibold">HuggingFace Address</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {[
                ['decidex-core-8b', 'LoRA adapter (flagship)', 'huggingface.co/fly88oj/decidex-core-8b'],
                ['decidex-true-8b', 'LoRA adapter (noul-perfect)', 'huggingface.co/fly88oj/decidex-true-8b'],
                ['decidex-draft-0.6b', 'MTP draft GGUF Q8_0', 'huggingface.co/fly88oj/decidex-draft-0.6b'],
                ['decidex-gguf', 'GGUF Q4/Q6/Q8 builds', 'huggingface.co/fly88oj/decidex-gguf'],
                ['decidex-distill-data', 'Distillation dataset (17,954)', 'huggingface.co/fly88oj/decidex-distill-data'],
              ].map(([name, type, url]) => (
                <tr key={name}>
                  <td className="px-3 py-3 md:px-5 font-medium">{name}</td>
                  <td className="px-3 py-3 md:px-5 text-muted-foreground">{type}</td>
                  <td className="px-3 py-3 md:px-5">
                    <a href={`https://${url}`} target="_blank" rel="noopener" className="text-blue-500 hover:underline break-all">
                      {url}
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  )
}
