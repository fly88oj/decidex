# Decidex

**Entra el estado. Salen decisiones tipadas. Un solo paso hacia adelante.**

Reimplementación local y de código abierto del modelo de decisiones Jev
(TypeSafe System One) y su API: sin generación de texto, sin análisis de
JSON, probabilidades calibradas en cada respuesta y entrenado mediante
destilación hacia las salidas reales del modelo oficial.

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | **Español** | [Français](README.fr.md) | [Deutsch](README.de.md)

> Documentación completa (guías, optimizaciones, comparación con la API
> oficial) en inglés: [README.md](README.md). Esta página resume lo esencial.

## ¿Qué es?

Decidex expone la misma API que Jev: envías un `state` (texto o JSON) y
preguntas tipadas, y recibes decisiones tipadas con probabilidades:

- **Choice** — elige una opción de hasta 255; devuelve `choice`,
  `probabilities`, `confidence`.
- **Score** — puntúa en 1–26 niveles ordenados (se recomiendan 2–10); el `score` puede caer entre
  niveles (`Σ i·pᵢ`).
- **Noul** — sí/no como probabilidad en [0, 1].

Todas las preguntas de una solicitud se evalúan en paralelo e
independientemente contra el mismo estado, con la fórmula oficial de
`confidence`. El contrato HTTP coincide campo por campo con el de la API
oficial (verificado con sus SDK de Python y JavaScript: basta cambiar la
URL base).

## Inicio rápido

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[all]"

# demo sin dependencias de ML
.venv/Scripts/python -m decidex demo --engine stub

# servicio (por defecto Qwen3-4B; descarga el modelo al primer uso)
HF_HOME=/camino/a/cache .venv/Scripts/python -m decidex serve --engine llm --port 8600
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

SDK de Python (interfaz equivalente a la oficial; solo cambia el import y la URL):

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
    print(r.answers["tone"].score)         # 0..2, puede caer entre niveles
```

## Niveles recomendados (medidos contra la API oficial)

| Nivel | Configuración | Latencia | Acuerdo con lo oficial |
|---|---|---|---|
| Latencia | `Qwen3-4B` | ~49 ms | noul 0.885, choice 1.000 |
| 16 GB de VRAM | `Qwen3-8B --dtype int4 --lora .../adapters/decidex-core-8b` | ~80 ms | noul 0.923 |
| Máximo acuerdo (24 GB) | `Qwen3-8B --lora .../adapters/decidex-core-8b` | ~64 ms | **total 84/86, choice 16/16** |

También hay compilaciones GGUF (Q4_K_M / Q6_K / Q8_0) con borrador MTP
destilado para llama.cpp / Ollama / LM Studio: [GGUF-USAGE.md](GGUF-USAGE.md).

## Enlaces

- Investigación y evidencia: [RESEARCH.md](RESEARCH.md) (inglés/chino)
- Comparación medida con la API oficial: [COMPARISON.md](COMPARISON.md)
- Registro de optimizaciones: [OPTIMIZATION.md](OPTIMIZATION.md)
- Panorama de proyectos de réplica: [REPLICATIONS.md](REPLICATIONS.md)
- Licencia MIT; contribuciones en [CONTRIBUTING.md](CONTRIBUTING.md)
