import { useMemo } from 'react'

interface Token {
  text: string
  cls: string
}

// Lightweight regex-based syntax highlighter for the code blocks.
// Supports: Python, Bash, JSON, and generic code.
function highlight(code: string): Token[] {
  const tokens: Token[] = []
  const patterns: [RegExp, string][] = [
    [/#[^\n]*/g, 'cm'],
    [/(["'])(?:\\.|[^\\\n])*?\1/g, 'str'],
    [/\b(?:from|import|def|return|with|as|for|in|if|else|elif|try|except|raise|class|pass|not|and|or|is|lambda|while|break|continue|print|export|const|let|var|async|await|new)\b/g, 'kw'],
    [/\b(?:True|False|None|true|false|null|undefined)\b/g, 'kw'],
    [/\b\d+\.?\d*\b/g, 'num'],
    [/[A-Z][a-zA-Z0-9]+(?=\s*\()/g, 'cls'],
    [/[a-zA-Z_][a-zA-Z0-9_]*(?=\s*\()/g, 'fn'],
    [/(?:==|!=|>=|<=|->|=>)/g, 'op'],
  ]

  let pos = 0
  while (pos < code.length) {
    let bestMatch: { index: number; length: number; cls: string } | null = null

    for (const [re, cls] of patterns) {
      re.lastIndex = pos
      const m = re.exec(code)
      if (m && m.index === pos) {
        if (!bestMatch || m[0].length > bestMatch.length) {
          bestMatch = { index: m.index, length: m[0].length, cls }
        }
      }
    }

    if (bestMatch) {
      tokens.push({ text: code.slice(pos, pos + bestMatch.length), cls: bestMatch.cls })
      pos += bestMatch.length
    } else {
      // Consume one unstyled character
      if (!tokens.length || tokens[tokens.length - 1].cls !== '') {
        tokens.push({ text: code[pos], cls: '' })
      } else {
        tokens[tokens.length - 1].text += code[pos]
      }
      pos++
    }
  }

  // Merge adjacent same-class tokens
  const merged: Token[] = []
  for (const t of tokens) {
    if (merged.length && merged[merged.length - 1].cls === t.cls) {
      merged[merged.length - 1].text += t.text
    } else {
      merged.push({ ...t })
    }
  }

  return merged
}

export function CodeBlock({ children }: { children: string }) {
  const tokens = useMemo(() => highlight(children), [children])

  return (
    <pre className="code-block">
      <code>
        {tokens.map((tok, i) =>
          tok.cls ? (
            <span key={i} className={tok.cls}>{tok.text}</span>
          ) : (
            <span key={i}>{tok.text}</span>
          )
        )}
      </code>
    </pre>
  )
}

