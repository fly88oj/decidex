import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { CheckCircle2, XCircle, AlertTriangle } from 'lucide-react'
import { CodeBlock } from '@/components/CodeBlock'

const contractItems = [
  { field: 'POST /v1/systemone', official: '{state, model, questions}', decidex: 'identical', status: 'pass' },
  { field: 'GET /v1/models', official: '{models: [{name, description, release_date}]}', decidex: 'identical', status: 'pass' },
  { field: '422 error body', official: '{detail: [{loc, msg, type}]} (FastAPI)', decidex: 'identical', status: 'pass' },
  { field: '401 / 429 / 529', official: 'official status codes + Retry-After', decidex: 'identical', status: 'pass' },
  { field: 'model aliases', official: 'jev-latest → jev-1.13.0', decidex: 'both accepted + decidex-latest', status: 'pass' },
  { field: 'score = Σ i·pᵢ', official: 'probability-weighted, between levels', decidex: 'exact formula match', status: 'pass' },
  { field: 'confidence formula', official: 'clamp((K·p_max-1)/(K-1), 0, 1)', decidex: 'exact formula match', status: 'pass' },
  { field: 'choice ≤ 255 options', official: 'max 255', decidex: 'enforced', status: 'pass' },
  { field: 'score levels', official: 'openapi floor 1, prose 2-10', decidex: '1-26 (letter capacity)', status: 'note' },
  { field: 'instructions optional', official: 'openapi: optional', decidex: 'optional (renders empty)', status: 'pass' },
  { field: 'noul criteria', official: '{true: str, false: str, both nullable}', decidex: 'accepted', status: 'pass' },
  { field: 'structured state', official: 'string | object | array', decidex: 'all three accepted', status: 'pass' },
]

const sdkTests = [
  { test: 'TypeSafeClient(api_key, base_url)', result: 'Python SDK 0.7.0', pass: true },
  { test: 'system_one() → typed answers', result: '.noul / .choice / .score all work', pass: true },
  { test: 'Grouped accessors', result: '.nouls / .choices / .scores', pass: true },
  { test: 'Dict-style questions', result: 'both dict and object forms', pass: true },
  { test: 'models.list()', result: 'ModelMetadataList shape', pass: true },
  { test: '422 error → TypeSafeUnprocessableEntityError', result: 'correct exception type', pass: true },
  { test: 'Env var redirect (TYPESAFE_BASE_URL)', result: 'zero code change', pass: true },
  { test: 'JS SDK (TypeSafeClient)', result: 'systemOne + models.list', pass: true },
]

export default function ApiCompat() {
  return (
    <main className="mx-auto max-w-6xl px-4 py-12 md:px-6">
      <h1 className="mb-2 text-4xl font-bold">Compatibility</h1>
      <p className="mb-8 text-lg text-muted-foreground">
        Already have code that calls the Jev API? Point it at localhost.
        That's the migration. No SDK fork, no code rewrite, no vendor lock-in.
      </p>

      {/* SDK verification */}
      <section className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Official SDK Verification</h2>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          {sdkTests.map(({ test, result, pass }) => (
            <div key={test} className="flex items-start gap-3 rounded-lg border p-4">
              {pass ? (
                <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-green-500" />
              ) : (
                <XCircle className="mt-0.5 h-5 w-5 shrink-0 text-red-500" />
              )}
              <div>
                <div className="font-medium">{test}</div>
                <div className="text-sm text-muted-foreground">{result}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Contract table */}
      <section className="mb-12">
        <h2 className="mb-4 text-2xl font-semibold">Contract Details</h2>
        <Card>
          <CardContent className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Field</TableHead>
                  <TableHead>Official Behavior</TableHead>
                  <TableHead>Decidex</TableHead>
                  <TableHead className="w-20">Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {contractItems.map(({ field, official, decidex, status }) => (
                  <TableRow key={field}>
                    <TableCell className="font-mono text-xs">{field}</TableCell>
                    <TableCell className="text-sm">{official}</TableCell>
                    <TableCell className="text-sm">{decidex}</TableCell>
                    <TableCell>
                      {status === 'pass' ? (
                        <CheckCircle2 className="h-4 w-4 text-green-500" />
                      ) : (
                        <AlertTriangle className="h-4 w-4 text-yellow-500" />
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </section>

      {/* Usage examples */}
      <section className="mb-12 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Python SDK (official)</CardTitle>
            <CardDescription>Only the import and base URL change</CardDescription>
          </CardHeader>
          <CardContent>
            <CodeBlock>
{`from typesafe_sdk import TypeSafeClient, Noul

client = TypeSafeClient(
    api_key="any",
    base_url="http://127.0.0.1:8600"
)
r = client.system_one(
    state="I was charged twice.",
    questions={
        "refund": Noul(instructions="Is this a refund request?")
    },
)
print(r.nouls["refund"].noul)  # 0..1`}
            </CodeBlock>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Decidex SDK (native)</CardTitle>
            <CardDescription>Same interface, better defaults</CardDescription>
          </CardHeader>
          <CardContent>
            <CodeBlock>
{`from decidex import DecidexClient, Noul

with DecidexClient() as client:  # defaults to localhost:8600
    r = client.system_one(
        state="I was charged twice.",
        questions={
            "refund": Noul(instructions="Is this a refund request?")
        },
    )
    print(r.answers["refund"].noul)`}
            </CodeBlock>
          </CardContent>
        </Card>
      </section>

      {/* Three primitives detail */}
      <section>
        <h2 className="mb-4 text-2xl font-semibold">The Three Primitives</h2>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          {[
            {
              name: 'Choice',
              criteria: 'map<option, description|null>, ≤255 entries',
              returns: 'choice + probabilities + confidence',
              example: `{"choice": "billing",\n "probabilities": {"billing": 0.99, ...},\n "confidence": 0.985}`,
            },
            {
              name: 'Score',
              criteria: 'array of 1-26 ordered level descriptions',
              returns: 'score + legend + probabilities + confidence',
              example: `{"score": 1.04,\n "legend": {"0": "Calm", ...},\n "probabilities": {"1": 0.95, ...}}`,
            },
            {
              name: 'Noul',
              criteria: 'optional {true: desc, false: desc}',
              returns: 'noul (float 0..1)',
              example: `{"noul": 0.95}`,
            },
          ].map(({ name, criteria, returns, example }) => (
            <Card key={name}>
              <CardHeader>
                <CardTitle className="font-mono">{name}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <Badge variant="secondary" className="mb-1">criteria</Badge>
                  <p className="text-xs text-muted-foreground">{criteria}</p>
                </div>
                <div>
                  <Badge variant="secondary" className="mb-1">returns</Badge>
                  <p className="text-xs text-muted-foreground">{returns}</p>
                </div>
                <code className="rounded bg-muted block p-2 text-[10px]">{example}</code>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>
    </main>
  )
}
