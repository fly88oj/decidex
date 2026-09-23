# Decidex

**ステートを入れ、型付きの決策を得る。1 回の forward pass。**

Jev（TypeSafe System One）決策モデル & API のローカル・オープン再実装 —
テキスト生成なし、パースなし、すべての回答に較正済み確率、公式モデルの
実際の出力への蒸留訓練付き。

[English](README.md) | [简体中文](README.zh-CN.md) | **日本語** | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md)

[調査ノート](RESEARCH.md) · 公式 API リファレンス: docs.typesafe.ai/api

**Decidex** は [TypeSafe AI の "Jev"](https://typesafe.ai/blog/introducing-system-one-models-and-jev) のオープンなローカル再実装です。
**非構造化ステートを入力すると、型付きの較正済み判断を出力します。**
テキスト生成も JSON 修復も幻覚フォーマットもありません。質問と回答は事前に定義した型の値であり、すべての質問は同じステートに対して**独立かつ並列**に評価され、1 回の呼び出しでミリ秒単位の応答が返ります。

```
┌─────────────┐   state（文字列 / JSON）    ┌──────────────────┐
│ あなたのコード │ ─────────────────────────► │  Decidex サービス │
│ (if/ルーティング)│  questions（Choice/Score/ │  ┌────────────┐ │
└─────────────┘   Noul、任意の数）           │  │ 凍結済み LM │ │ 1 回の
       ▲                                    │  │ 直接 logits │ │ forward pass
       │   answers: 型付き値 + 確率          │  └────────────┘ │ で確率を読む
       └────────────────────────────────────└──────────────────┘
```

## クイックスタート

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[all]"

# 依存ゼロのスモークテスト（語彙的スタブエンジン）
.venv/Scripts/python -m decidex demo --engine stub

# サービス起動（デフォルト: Qwen3-4B 直接 logits 読み出し。初回実行時に
# モデルをダウンロード。HF_HOME を大容量ドライブに指定してください）
HF_HOME=/path/to/hf-cache .venv/Scripts/python -m decidex serve --engine llm --port 8600
```

呼び出し例 — 公式 API とフィールド単位で完全一致:

```bash
curl -s http://127.0.0.1:8600/v1/systemone -H "Content-Type: application/json" -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "decidex-latest",
  "questions": {
    "is_urgent":   {"type": "noul",   "instructions": "Does this convey urgency?"},
    "department":  {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"billing": "Payments, invoicing, refunds",
                                 "technical": "Bugs, outages, integrations",
                                 "sales": "Pricing, upgrades, new accounts"}},
    "frustration": {"type": "score",  "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  }
}'
```

Python SDK — 公式 `typesafe_sdk` とインターフェース互換。import と
ベース URL を変えるだけで、既存コードがローカルサーバーに対して動きます:

```python
from decidex import Choice, DecidexClient, Noul, Score

with DecidexClient() as client:          # http://127.0.0.1:8600
    response = client.system_one(
        state={"message": "I was charged twice for order A-104.", "order": {"id": "A-104"}},
        questions={
            "refund_requested": Noul("Does `message` request a refund?"),
            "request_type": Choice("What is the main request in `message`?",
                                   criteria={"refund": "Wants money returned",
                                             "information": "Asking a question",
                                             "other": "Anything else"}),
            "frustration": Score("How frustrated is the customer?",
                                 criteria=["Calm", "Frustrated", "Very angry"]),
        },
    )
    print(response.answers["refund_requested"].noul)      # 0..1
    print(response.answers["request_type"].choice)        # "refund"
    print(response.answers["request_type"].confidence)    # 0..1
    print(response.answers["frustration"].score)          # 0..2、レベル間にも落下可
```

完全な動作例（confidence ゲート付きルーティングのチケットトリアージ、
公式 intent-routing パターン）: [`examples/ticket_triage.py`](examples/ticket_triage.py)

## 3 つのプリミティブ

| 型 | criteria | 返り値 | 意味 |
|---|---|---|---|
| `choice` | `map<選択肢, 説明|null>`、最大 255 項目 | `choice` + `probabilities` + `confidence` | 選択 = 確率の argmax |
| `score` | 1–26 個の順序付きレベル記述（公式は 2–10 を推奨） | `score` + `legend` + `probabilities` + `confidence` | `score = Σ i·p_i`（レベル間にも落下可） |
| `noul` | 任意の `{true, false}` 記述 | `noul` ∈ [0,1] | 「はい」の確率。値そのものが信念のため confidence なし |

`confidence = clamp((K·p_max − 1)/(K − 1), 0, 1)` — 公式ドキュメントの
インタラクティブデモのソースに公開されている式で、公式 API リファレンスの
2 つの実例で検証済みです（[RESEARCH.md](RESEARCH.md) 参照）。

## 公式 API との互換性

**実機検証済みのドロップイン。** 両公式 SDK がベース URL の変更だけでローカルサーバーに対して動作します:

- **Python** `typesafe-sdk`（0.7.0 で検証）: `TypeSafeClient(api_key="x", base_url="http://127.0.0.1:8600")`、または `TYPESAFE_BASE_URL` + `TYPESAFE_API_KEY` 環境変数で**コード零変更**。公式ドキュメントの例がそのまま動作し、`.nouls`/`.choices`/`.scores` のグループ別アクセサ、dict 形式の質問、null 選択肢記述、`client.models.list()` にも対応。
- **JavaScript/TypeScript** `@typesafe-ai/sdk`（0.6.0 で検証）: `new TypeSafeClient({ apiKey: "x", baseURL: "http://127.0.0.1:8600" })` —— `systemOne` / `models.list` / 型付き回答がすべて通過。
- 生 HTTP も公式形状と一致。422 は FastAPI 風の `{"detail": [{"loc", "msg", "type"}]}`（公式契約は `typesafe-sdk` に OpenAPI 生成 schema として同梱されており、Decidex はそれに従います）。

公式 `openapi.json` と整合した詳細:

- `POST /v1/systemone`、`GET /v1/models` のパス・フィールドが一致（`model`/`answers`/`usage`）。
- `GET /v1/models` は公式 `ModelMetadataList` 形状: `{models: [{name, description, release_date}]}`。
- エラー意味論: `401`、`422`（FastAPI 風 detail が問題フィールドを明示）、`429`/`529`（リトライ可。SDK は指数バックオフ + `Retry-After` 尊重）。
- `instructions` は 3 種類の質問型すべてで省略可。noul の `criteria.true/false` は null を受容。score の `legend` は criteria をそのまま返す（構造化値を含む）。
- モデル名: `decidex-latest`/`decidex-1.1.0` と公式の `jev-latest`/`jev-1.13.0`。
- 上限: choice は 255 選択肢まで。score は 1–26 レベル（公式 openapi の下限は 1、ドキュメント本文は 2–10 を推奨。26 は文字読み出しの上限で、超過時は明確な 422 を返す）。

検証を自分で再実行するには（サーバー起動中かつ `pip install -e ".[compat]"` が必要）:

```bash
.venv/Scripts/python tests/check_official_sdk_compat.py
```


## アーキテクチャ

```
decidex/
├── server.py            FastAPI: /v1/systemone の検証 + 並列評価 + 公式レスポンス形状
├── sdk.py               DecidexClient + Choice/Score/Noul（typesafe_sdk と同一形状）
├── calib.py             公式の数式: softmax、confidence、Σi·p_i、確率の総和 = 1
├── render.py            state/instructions/criteria → テキスト（文字列|オブジェクト|配列）
├── cli.py               python -m decidex serve | demo
└── engines/
    ├── llm_logits.py    ★ デフォルトエンジン: 凍結 LM から 1 回の forward pass で
    │                       選択肢文字の logits を直接読む（SemIf のアプローチ）
    ├── embedding.py     代替: sentence-transformer のコサイン類似度（CPU で動く、ヒューリスティック）
    └── base.py          Engine インターフェース + 依存ゼロの語彙的スタブ（テスト用）
```

**LLM 直接 logits エンジン**（再実装の核）: 選択肢を `A./B./C.…` とラベル付けし、
1 回の forward pass で最終位置の選択肢文字 logits を読んで softmax で確率化します。
**トークンを 1 つもサンプルせず、何も生成しない**——これが Jev の「並列サンプラー・
出力無料」の力学的等価物です（コミュニティ再実装 SemIf は同じ方式で出力 0 トークン、
自己回帰 JSON 比 5.2 倍速を計測）。1 リクエスト内の全質問はステートを共有し、
パディングされた並列バッチでスコア化します。26 選択肢を超えると選択肢ごとの
関連性プローブ + 再正規化に切り替わります——公式が高基数 Choice で説明するのと
同じ 2 段階構成です。

## 開発機での実測値（RTX 5080 / Qwen3-4B）

| シナリオ | 最適化前 | 最適化後 |
|---|---|---|
| 質問 1 件（ウォーム p50） | 53 ms | **49 ms** |
| 10 質問並列ファンアウト（ウォーム） | 151 ms | **66 ms** |
| 3.3k トークン文書 × 10 質問（コールド） | OOM クラッシュ | **2.2 s** |
| 同一文書の再問い合わせ（プレフィックスキャッシュ） | — | **193 ms** |
| 32 並発リクエスト | — | すべて成功、負荷中 /health p50 2ms、5xx ゼロ |

53 項目ベンチ（皮肉・二重否定・近似選択肢の難例含む）: noul 24/24、choice
16/16（較正 gap −0.0009）、score 難例 84.6%（`ensemble_rounds=3` で 100%）。
最適化の全記録は [OPTIMIZATION.md](OPTIMIZATION.md)（英語）。

## 推奨ティア（公式 API 対して実測）

| ティア | 構成 | レイテンシ | 公式との一致率 |
|---|---|---|---|
| 低遅延（任意 GPU） | `Qwen3-4B` ベースモデル | ~49 ms | noul 決定 0.885、choice 満点 |
| 16GB | `Qwen3-8B --dtype int4 --lora benchmarks/adapters/decidex-core-8b` | ~80 ms | noul 決定 0.923 |
| 最高一致（24GB） | `Qwen3-8B --lora benchmarks/adapters/decidex-core-8b` | ~64 ms | **合計 84/86、choice 満点** |

アダプターの意味的命名: **core-8b**（デフォルト推奨・合計最高）/
**true-8b**（noul 満点 52/52 版、`benchmarks/adapters/decidex-true-8b`）/
**draft-0.6b**（MTP 投機デコード用ドラフト）。
GGUF 版は血統マーカー付き: `decidex-core-8b-r1-Q4_K_M.gguf`（r1 = core 系第 1 版）。

## 設定

| 環境変数 / フラグ | デフォルト | 備考 |
|---|---|---|
| `--engine` | `llm`（serve）/ `stub`（demo） | `llm` \| `embedding` \| `stub` |
| `--model` | エンジン毎のデフォルト（Qwen/Qwen3-4B / all-MiniLM-L6-v2） | HF モデル名またはローカルパス。任意の causal LM が動作 |
| `--temperature` | 1.0（llm）/ 0.05（embedding） | 確率の鋭さ。過信を緩和するには上げる |
| `--device` / `DECIDEX_DEVICE` | 自動 cuda→cpu | マルチ GPU 環境で `cuda:1` などを選択 |
| `--api-key` / `DECIDEX_API_KEY` | なし（ローカルは無認証） | 設定すると Bearer 認証を強制 |
| `DECIDEX_BASE_URL` | `http://127.0.0.1:8600` | SDK のデフォルト接続先 |
| `DECIDEX_MAX_INPUT_TOKENS` | 65536 | サーバー側コンテキスト上限（llm エンジンはさらに 8192 を強制、小さい方が優先） |
| `DECIDEX_PREFIX_REUSE` | 1 | KV プレフィックス再利用: 共有 state の forward を 1 回に |
| `DECIDEX_PREFIX_CACHE_GB` | 4 | リクエスト横断プレフィックス KV キャッシュの LRU 予算（GB）、0 で無効 |
| `DECIDEX_MAX_QUEUE` | 8 | 同時待機評価の上限。超過時は 429 + `Retry-After` |
| `HF_HOME` | `~/.cache/huggingface` | モデルキャッシュ。大容量ドライブを推奨 |

> ネットワークのヒント: PyPI が遅いネットワークでは
> `-i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir` でインストールしてください。

## 公式 Jev との違い（正直なリスト）

| 観点 | 公式 Jev | Decidex |
|---|---|---|
| モデル | 非公開の自社アーキテクチャ | 凍結されたオープン LM（デフォルト Qwen3-4B。`--model` で任意の causal LM に変更可） |
| 学習 | RLCD（較正済み判断に対する強化学習） | なし。確率は logits + 温度パラメータから |
| 較正 | 公式に較正された answer 毎の confidence | 単一の温度ノブ。式は一致するが較正品質は基底モデル次第 |
| レイテンシ | 70–500 ms（専用サービス） | noul 1 件 53 ms / 10 質問 130 ms（5080 + 4B で実測） |
| コンテキスト | 64k トークン | llm エンジンはデフォルト 8k（`max_input_tokens`）、サーバー上限 64k |
| トークン計数 | 正確 | エンジンがトークナイザを持てば正確、それ以外は chars/4 推定 |

## テスト

```bash
.venv/Scripts/python -m pytest tests -m "not slow"   # 契約+数式+SDK、ML 依存なし、数秒
.venv/Scripts/python -m pytest tests -m slow          # 実エンジンのスモークテスト（ML スタック必須）
```

## 謝辞とライセンス

- 公開情報に基づいて構築: TypeSafe AI のドキュメントとブログ（API 形状・数式）、
  およびコミュニティ再実装 [SemIf](https://github.com/TheoLeeCJ/SemIf)（直接 logits 方式）。
  本プロジェクトは TypeSafe AI と無関係です。Jev と TypeSafe は各所有者の商標です。
- プロジェクトコード: MIT（[LICENSE](LICENSE) 参照）。貢献: [CONTRIBUTING.md](CONTRIBUTING.md)。変更: [CHANGELOG.md](CHANGELOG.md)。
