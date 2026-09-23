# Decidex

**状态进，类型化决策出，一次前向传播。**

Jev（TypeSafe System One）决策模型与 API 的本地开源复刻——无文本生成、
无解析、每个答案都带校准概率，并朝官方模型的真实输出蒸馏训练。

[English](README.md) | **简体中文** | [日本語](README.ja.md) | [Español](README.es.md) | [Français](README.fr.md) | [Deutsch](README.de.md)

[调研文档](RESEARCH.zh-CN.md) | 官方 API 参考：docs.typesafe.ai/api

**Decidex** 是 [TypeSafe AI "Jev"](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 的本地开源复刻：**非结构化状态进（state in），类型化的校准决策出（typed decisions out）**。没有文本生成、没有 JSON 解析、没有幻觉格式——问题和答案都是你在请求里预先定义的类型，所有问题对同一状态**并行独立评估**，一次调用毫秒级返回。

```
┌─────────────┐   state（字符串/JSON）      ┌──────────────────┐
│ 你的代码     │ ─────────────────────────► │  Decidex 服务     │
│ （if/路由）  │   questions（Choice/Score/  │  ┌────────────┐  │
└─────────────┘   Noul，任意多个）           │  │ 冻结开源LM  │  │ 一次前向
       ▲                                    │  │ 直接logits  │  │ 读出概率
       │   answers: 类型值 + probabilities   │  └────────────┘  │
       └────────────────────────────────────└──────────────────┘
```

## 快速开始

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[all]"

# 无 ML 栈冒烟（词法 stub 引擎）
.venv/Scripts/python -m decidex demo --engine stub

# 启动服务（默认 Qwen3-4B 直接 logits 读出，需 GPU；首次运行自动下载模型）
.venv/Scripts/python -m decidex serve --engine llm --port 8600
```

调用（与官方 API 逐字段一致，见下方对照）：

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

Python SDK（接口对齐官方 `typesafe_sdk`，只换 import 和 base_url）：

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
    print(response.answers["frustration"].score)          # 0..2，可落在等级之间
```

完整示例（工单分类 + 置信度门控路由，官方 intent-routing 模式）：[`examples/ticket_triage.py`](examples/ticket_triage.py)

## 三种原语

| 类型 | criteria | 返回 | 语义 |
|---|---|---|---|
| `choice` | `map<选项, 描述|null>`，≤255 项 | `choice` + `probabilities` + `confidence` | 选中项 = 概率 argmax |
| `score` | 1–26 个有序等级描述（官方建议 2–10） | `score` + `legend` + `probabilities` + `confidence` | `score = Σ i·p_i`（可落在等级之间） |
| `noul` | 可选 `{true, false}` 描述 | `noul` ∈ [0,1] | "是"的概率；数值本身就是信念，无 confidence |

`confidence = clamp((K·p_max − 1)/(K − 1), 0, 1)`——公式直接取自官方文档交互演示的源码（见 [RESEARCH.zh-CN.md](RESEARCH.zh-CN.md#12-三种原语api-的全部)），并用官方 API 参考的两个示例数值验证过。

## 与官方 API 的兼容性

**已实测平替。** 两个官方 SDK 只改 base URL 即可直连本服务，无需其它代码改动：

- **Python** `typesafe-sdk`（0.7.0 实测）：`TypeSafeClient(api_key="x", base_url="http://127.0.0.1:8600")`，或用 `TYPESAFE_BASE_URL` + `TYPESAFE_API_KEY` 环境变量实现**零代码改动**。官方文档示例原样可跑，含 `.nouls`/`.choices`/`.scores` 分组访问器、字典问题、null 选项描述、`client.models.list()`。
- **JavaScript/TypeScript** `@typesafe-ai/sdk`（0.6.0 实测）：`new TypeSafeClient({ apiKey: "x", baseURL: "http://127.0.0.1:8600" })`——`systemOne`/`models.list`/类型化答案全部通过。
- 裸 HTTP 同样对齐官方形状，包括 422 的 FastAPI 风格 `{"detail": [{"loc", "msg", "type"}]}`（官方契约内嵌于 `typesafe-sdk` 的 OpenAPI 生成 schema，Decidex 按其实现）。

与官方 `openapi.json` 对齐的细节：

- `POST /v1/systemone`、`GET /v1/models` 路径与字段一致（`model`/`answers`/`usage`）。
- `GET /v1/models` 返回官方 `ModelMetadataList` 形状：`{models: [{name, description, release_date}]}`。
- 错误语义：`401`、`422`（FastAPI 风格 detail 指明出错字段）、`429`/`529`（可重试；SDK 指数退避并尊重 `Retry-After`）。
- `instructions` 三种问题均可选；noul 的 `criteria.true/false` 接受 null；score 的 `legend` 原样回显 criteria（含结构化值）。
- 模型名：`decidex-latest`/`decidex-1.1.0` 与官方 `jev-latest`/`jev-1.13.0`。
- 校验上限：choice ≤255 选项；score 接受 1–26 级（官方 openapi 下限为 1，文档建议 2–10；26 为字母读出上限，超出返回明确 422）。

自行复验（需服务已启动并 `pip install -e ".[compat]"`）：

```bash
.venv/Scripts/python tests/check_official_sdk_compat.py
```

## 架构

```
decidex/
├── server.py            FastAPI：/v1/systemone 校验 + 并行评估 + 官方响应形状
├── sdk.py               DecidexClient + Choice/Score/Noul（对齐 typesafe_sdk）
├── calib.py             官方公式：softmax、confidence、Σi·p_i、概率和恒为 1
├── render.py            state/instructions/criteria → 文本（str|object|array）
├── cli.py               python -m decidex serve | demo
└── engines/
    ├── llm_logits.py    ★ 默认引擎：冻结 LLM 单次前向直接读选项字母 logits（SemIf 方案）
    ├── embedding.py     备选：sentence-transformers 余弦相似度（CPU 可跑，启发式）
    └── base.py          Engine 接口 + 无依赖 stub（词法重叠，供测试）
```

**LLM logits 引擎**（复刻核心）：选项以 `A./B./C.…` 标注进 prompt，一次前向传播读取末位对字母 token 的 logits，softmax 成概率——**不采样任何 token、不生成任何文本**，这正是 Jev "并行采样、输出免费" 的机制等价物（社区复刻 SemIf 用同方案实测：0 输出 token，比自回归生成 JSON 快 5.2 倍）。同一请求的所有问题共享 state，分批并行前向；超过 26 个选项自动切换为独立相关性探针 + 归一化（官方对高基数同样采用两阶段）。

## 与官方 Jev 的差异（诚实清单）

| 方面 | 官方 Jev | Decidex |
|---|---|---|
| 模型 | 未公开的自研架构 | 冻结的开源 LLM（默认 Qwen3-4B，`--model` 可换任意因果 LM） |
| 训练 | RLCD（校准决策强化学习） | 无训练；概率来自 logits + 温度（`--temperature` 校准旋钮） |
| 校准程度 | 官方称每答必带校准 confidence | 温度单一旋钮，置信度公式与官方一致但校准质量取决于底座模型 |
| 延迟 | 70–500ms（专用服务） | 单 noul 53ms / 10 问题 130ms（5080+4B 实测） |
| 上下文 | 64k token | 默认 8k（`DECIDEX_MAX_INPUT_TOKENS` 可调，受底座模型限制） |
| token 计数 | 精确 | 引擎有 tokenizer 时精确，否则 chars/4 估算 |

## 实测（RTX 5080 / Qwen3-4B / 本仓库开发机）

| 场景 | 优化前 | 优化后 |
|---|---|---|
| 单个问题（热 p50） | 53 ms | **49 ms** |
| 10 问题并行 fan-out（热） | 151 ms | **66 ms** |
| 3.3k token 长文档 × 10 问题（冷） | OOM 崩溃 | **2.2 s** |
| 同一文档再次查询（前缀缓存命中） | — | **193 ms** |
| 32 并发请求 | — | 全部成功，负载中 /health p50 2ms，零 5xx |

53 项基准（含讽刺/双重否定/近似选项难例）：noul 24/24、choice 16/16（校准
gap −0.0009）、score 难例 84.6%（开启 `ensemble_rounds=3` 后 100%）。每项
优化与否决都有实测记录：[OPTIMIZATION.md](OPTIMIZATION.md)。

## 推荐档位（对官方实测）

| 档位 | 配置 | 延迟 | 对官方一致率 |
|---|---|---|---|
| 延迟档（任意卡） | `Qwen3-4B` 裸模型 | ~49 ms | noul 决策 0.885、choice 满分 |
| 16GB 档 | `Qwen3-8B --dtype int4 --lora benchmarks/adapters/decidex-core-8b` | ~80 ms | noul 决策 0.923 |
| 最佳一致档（24GB） | `Qwen3-8B --lora benchmarks/adapters/decidex-core-8b` | ~64 ms | **总分 84/86、choice 满分** |

Adapter 语义命名：**core-8b**（默认推荐，总分最高）/ **true-8b**（noul 满分 52/52 变体，
`benchmarks/adapters/decidex-true-8b`）/ **draft-0.6b**（MTP 投机解码伴生模型）。
GGUF 量化版带血统标志：`decidex-core-8b-r1-Q4_K_M.gguf`（r1 = core 血统第一版）。

质量抽查（官方 API 参考的同款请求）：官方 department 概率 `{billing 0.88, technical 0.12, sales 0.0}`、confidence 0.81；本服务输出 `{0.87, 0.13, 0.0005}`、confidence 0.80。工单分流示例（examples/ticket_triage.py）三张工单全部正确路由：愤怒的重复扣款 → `auto_refund_flow`，外观 bug → `bug_backlog`，登录问题 → `access_support_flow`。

## 配置

| 环境变量 / 参数 | 默认 | 说明 |
|---|---|---|
| `--engine` | `llm`（serve）/`stub`（demo） | `llm` \| `embedding` \| `stub` |
| `--model` | 引擎各自默认（Qwen/Qwen3-4B / all-MiniLM-L6-v2） | HF 模型名或本地路径；任何因果 LM 均可 |
| `--temperature` | 1.0（llm）/ 0.05（embedding） | 概率锐度；调大可缓解过度自信（4B 原始 logits 偏尖锐） |
| `--device` / `DECIDEX_DEVICE` | 自动 cuda→cpu | 多卡机器指定 `cuda:1` 等（开发机上 cuda:0=4090 另有任务，本项目用 cuda:1=5080） |
| `--api-key` / `DECIDEX_API_KEY` | 无（本地免鉴权） | 设置后强制 Bearer 校验 |
| `DECIDEX_BASE_URL` | `http://127.0.0.1:8600` | SDK 默认地址 |
| `DECIDEX_MAX_INPUT_TOKENS` | 65536 | 服务端上下文上限（llm 引擎另受自身 8192 限制，取二者较小值） |
| `DECIDEX_PREFIX_REUSE` | 1 | KV 前缀复用：共享 state 只前向一次 |
| `DECIDEX_PREFIX_CACHE_GB` | 4 | 跨请求 state 前缀 KV 缓存的 LRU 预算（GB），0 关闭 |
| `DECIDEX_MAX_QUEUE` | 8 | 最多并发等待的评估数，超出返回 429 + `Retry-After` |
| `HF_HOME` | `~/.cache/huggingface` | 模型缓存位置（建议指到大容量盘，如 `E:/Works/hf-cache`） |

> 网络提示：本机直连 PyPI 极慢且 pip 缓存易损坏，装依赖建议 `-i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir`；HuggingFace 直连可用。

## 测试

```bash
.venv/Scripts/python -m pytest tests -m "not slow"   # 契约+数学+SDK（无 ML 依赖，秒级）
.venv/Scripts/python -m pytest tests -m slow          # 引擎冒烟（需已安装 ML 栈与模型）
```

## 致谢与许可

- 实现依据来自 TypeSafe AI 公开文档与博客（API 形状、公式）及社区复刻 [SemIf](https://github.com/TheoLeeCJ/SemIf)（直接 logits 方案）。本项目与 TypeSafe AI 无关联；Jev、TypeSafe 为其各自所有者的商标。
- 本项目代码 MIT（见 [LICENSE](LICENSE)）。参与贡献：[CONTRIBUTING.md](CONTRIBUTING.md)。变更记录：[CHANGELOG.md](CHANGELOG.md)。
