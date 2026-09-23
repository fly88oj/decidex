# Decidex

**State rein. Typisierte Entscheidungen raus. Ein einziger Forward-Pass.**

Lokale Open-Source-Nachbildung des Entscheidungsmodells Jev (TypeSafe
System One) samt API: keine Textgenerierung, kein JSON-Parsing, kalibrierte
Wahrscheinlichkeiten bei jeder Antwort — trainiert per Destillation auf die
tatsächlichen Ausgaben des offiziellen Modells.

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | **Deutsch**

> Die vollständige Dokumentation (Anleitungen, Optimierungen, Vergleich mit
> der offiziellen API) liegt auf Englisch: [README.md](README.md). Diese
> Seite fasst das Wesentliche zusammen.

## Was ist das?

Decidex bietet dieselbe API wie Jev: Sie senden einen `state` (Text oder
JSON) und typisierte Fragen und erhalten typisierte Entscheidungen mit
Wahrscheinlichkeiten:

- **Choice** — wählt eine von bis zu 255 Optionen; liefert `choice`,
  `probabilities`, `confidence`.
- **Score** — bewertet auf 1–26 geordneten Stufen (2–10 empfohlen); der `score` kann
  zwischen Stufen liegen (`Σ i·pᵢ`).
- **Noul** — Ja/Nein als Wahrscheinlichkeit in [0, 1].

Alle Fragen einer Anfrage werden parallel und unabhängig gegen denselben
State ausgewertet, mit der offiziellen `confidence`-Formel. Der
HTTP-Vertrag stimmt Feld für Feld mit der offiziellen API überein
(verifiziert mit deren Python- und JavaScript-SDKs: nur die Base-URL
ändert sich).

## Schnellstart

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[all]"

# Demo ohne ML-Abhängigkeiten
.venv/Scripts/python -m decidex demo --engine stub

# Dienst (Standard Qwen3-4B; lädt das Modell beim ersten Start)
HF_HOME=/pfad/zum/cache .venv/Scripts/python -m decidex serve --engine llm --port 8600
```

```bash
curl -s http://127.0.0.1:8600/v1/systemone -H "Content-Type: application/json" -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "decidex-latest",
  "questions": {
    "is_urgent":  {"type": "noul", "instructions": "Does this convey urgency?"},
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"billing": "Payments, invoicing, refunds",
                                "technical": "Bugs, outages, integrations",
                                "sales": "Pricing, upgrades, new accounts"}},
    "frustration":{"type": "score", "instructions": "How frustrated is the customer?",
                   "criteria": ["Calm", "Frustrated", "Very angry"]}
  }
}'
```

Python-SDK (Schnittstelle wie das offizielle; nur der Import ändert sich):

```python
from decidex import Choice, DecidexClient, Noul, Score

with DecidexClient() as client:
    r = client.system_one(
        state="I was charged twice for order A-104.",
        questions={"refund": Noul("Does this request a refund?"),
                   "tone": Score("How frustrated is the customer?",
                                 criteria=["Calm", "Frustrated", "Very angry"])},
    )
    print(r.answers["refund"].noul)        # 0..1
    print(r.answers["tone"].score)         # 0..2, kann zwischen Stufen liegen
```

## Empfohlene Stufen (gegen die offizielle API gemessen)

| Stufe | Konfiguration | Latenz | Übereinstimmung |
|---|---|---|---|
| Latenz | `Qwen3-4B` | ~49 ms | noul 0,885, choice 1,000 |
| 16 GB VRAM | `Qwen3-8B --dtype int4 --lora .../adapters/decidex-core-8b` | ~80 ms | noul 0,923 |
| Max. Übereinstimmung (24 GB) | `Qwen3-8B --lora .../adapters/decidex-core-8b` | ~64 ms | **gesamt 84/86, choice 16/16** |

GGUF-Builds (Q4_K_M / Q6_K / Q8_0) mit destilliertem MTP-Draft für
llama.cpp / Ollama / LM Studio: [GGUF-USAGE.md](GGUF-USAGE.md).

## Links

- Forschung & Belege: [RESEARCH.md](RESEARCH.md) (Englisch/Chinesisch)
- Gemessener Vergleich mit der offiziellen API: [COMPARISON.md](COMPARISON.md)
- Optimierungs-Journal: [OPTIMIZATION.md](OPTIMIZATION.md)
- Replikations-Landschaft: [REPLICATIONS.md](REPLICATIONS.md)
- MIT-Lizenz; Beiträge: [CONTRIBUTING.md](CONTRIBUTING.md)
