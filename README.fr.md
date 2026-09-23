# Decidex

**L'état entre. Des décisions typées sortent. Une seule passe avant.**

Réimplémentation locale et open source du modèle de décision Jev
(TypeSafe System One) et de son API : pas de génération de texte, pas
d'analyse JSON, des probabilités calibrées avec chaque réponse, et un
entraînement par distillation vers les sorties réelles du modèle officiel.

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [Español](README.es.md) | **Français** | [Deutsch](README.de.md)

> Documentation complète (guides, optimisations, comparaison avec l'API
> officielle) en anglais : [README.md](README.md). Cette page résume
> l'essentiel.

## Qu'est-ce que c'est ?

Decidex expose la même API que Jev : vous envoyez un `state` (texte ou
JSON) et des questions typées ; vous recevez des décisions typées avec
probabilités :

- **Choice** — choisit une option parmi 255 ; renvoie `choice`,
  `probabilities`, `confidence`.
- **Score** — évalue sur 1 à 26 niveaux ordonnés (2 à 10 recommandés) ; le `score` peut tomber
  entre deux niveaux (`Σ i·pᵢ`).
- **Noul** — oui/non sous forme de probabilité dans [0, 1].

Toutes les questions d'une requête sont évaluées en parallèle et
indépendamment sur le même état, avec la formule officielle de
`confidence`. Le contrat HTTP correspond champ par champ à celui de l'API
officielle (vérifié avec ses SDK Python et JavaScript : seul change l'URL
de base).

## Démarrage rapide

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[all]"

# démo sans dépendances ML
.venv/Scripts/python -m decidex demo --engine stub

# service (par défaut Qwen3-4B ; télécharge le modèle au premier usage)
HF_HOME=/chemin/vers/cache .venv/Scripts/python -m decidex serve --engine llm --port 8600
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

SDK Python (interface identique à l'officiel ; seul l'import change) :

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
    print(r.answers["tone"].score)         # 0..2, peut tomber entre niveaux
```

## Niveaux recommandés (mesurés contre l'API officielle)

| Niveau | Configuration | Latence | Accord avec l'officiel |
|---|---|---|---|
| Latence | `Qwen3-4B` | ~49 ms | noul 0.885, choice 1.000 |
| 16 Go de VRAM | `Qwen3-8B --dtype int4 --lora .../adapters/decidex-core-8b` | ~80 ms | noul 0.923 |
| Accord max (24 Go) | `Qwen3-8B --lora .../adapters/decidex-core-8b` | ~64 ms | **total 84/86, choice 16/16** |

Des builds GGUF (Q4_K_M / Q6_K / Q8_0) avec brouillon MTP distillé
existent pour llama.cpp / Ollama / LM Studio : [GGUF-USAGE.md](GGUF-USAGE.md).

## Liens

- Recherche et preuves : [RESEARCH.md](RESEARCH.md) (anglais/chinois)
- Comparaison mesurée avec l'API officielle : [COMPARISON.md](COMPARISON.md)
- Journal des optimisations : [OPTIMIZATION.md](OPTIMIZATION.md)
- Panorama des projets de réplication : [REPLICATIONS.md](REPLICATIONS.md)
- Licence MIT ; contributions dans [CONTRIBUTING.md](CONTRIBUTING.md)
