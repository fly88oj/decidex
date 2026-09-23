# Jev 实现依据调研

（[English version](RESEARCH.md)）

> 调研日期：2026-09-21。目标：弄清 TypeSafe AI **Jev**（System One 模型）的功能与接口依据、社区复刻方案，作为本仓库 `Decidex` 本地复刻的设计输入。

## 1. Jev 是什么（官方依据）

来源：[TypeSafe AI 官方博客《Introducing System One Models & Jev》(2026-09-15)](https://typesafe.ai/blog/introducing-system-one-models-and-jev)、[官方文档 docs.typesafe.ai](https://docs.typesafe.ai/)。

- **发布**：2026-09-15，TypeSafe AI（创始人 Diogo Almeida，OpenAI 前 InstructGPT/RLHF 共同作者），early access。
- **定位**：第一个 "System One" 模型 —— 取自 Kahneman《思考，快与慢》中的"快思考"。名字 Jev 来自经济学家 William Stanley Jevons（Jevons 悖论：效率提升→用量暴涨）。
- **一句话**：*"frontier-intelligence function call: unstructured state in, typed probabilistic decisions out"* —— 非结构化状态进，类型化概率决策出。
- **它不是聊天模型**：完全放弃文本生成（string generation），因此**按构造不可能产生类型错误/幻觉格式**（但可以答错——"wrong valid value"）。

### 1.1 与 LLM 的机制差异（官方对照表摘要）

| 维度 | 传统 LLM | Jev (System One) |
|---|---|---|
| 训练 | RLHF / RLVR | **RLCD**（Reinforcement Learning for Calibrated Decisions，直接对"校准决策"做强化学习） |
| 输入 | 非结构化文本，强调消息序列 | 非结构化数据，**强调结构化程序状态** |
| 输出 | 字符串（需要解析+校验） | **类型安全结构值 + 校准概率 + confidence** |
| 采样 | 自回归，逐 token | **并行**：一次前向出全部答案，输出侧接近零成本 |
| 速度 | 3–329 s | 70–500 ms |
| 价格 | 输入 $0.20–10/MTok，输出约 5× 输入 | 输入 $0.042/MTok，输出免费 |
| 置信度 | 过度自信、不一致 | 每个答案都带 confidence，校准（高 confidence ⇒ 高准确率） |

### 1.2 三种原语（API 的全部）

官方文档 [Primitives](https://docs.typesafe.ai/primitives) / [API 参考](https://docs.typesafe.ai/api)：

| 问题类型 | 问什么 | criteria | 返回 |
|---|---|---|---|
| **Choice** | 从集合中选一个 | `map<option, 描述|null>`，**最多 255 个选项** | `choice`（argmax 选项）、`probabilities`（每选项一个，和为 1）、`confidence` |
| **Score** | 按等级量表打分 | 有序数组，**2–10 个等级描述** | `score`（概率加权位置，可落在等级之间，如 `1.05`）、`legend`、`probabilities`、`confidence` |
| **Noul** | 陈述是否为真 | 可选 `{true: 描述, false: 描述}` | `noul` ∈ [0,1]（"是"的概率；无 confidence——数值本身就是信念） |

语义要点（官方原文确认）：

- `score` 是**概率加权的等级下标**：`score = Σ p_i · i`。官方示例 `{0:0.0, 1:0.95, 2:0.05} → score 1.05`，验算 `0×0+1×0.95+2×0.05 = 1.05` ✓
- `confidence` 由概率分布**推导**。官方文档交互演示给出了确切公式（JS 源码）：
  ```js
  confidence = clamp((K · p_max − 1) / (K − 1), 0, 1)   // K = 选项/等级数
  ```
  即"超出均匀分布的过剩确信度"。用官方两个示例验算：K=3, p_max=0.88 → 0.82（官方 0.81）；p_max=0.95 → 0.925（官方 0.92）✓
- 所有问题在**同一次调用中并行、独立**评估，共享同一 `state`；加问题几乎不增加延迟（speculative fan-out 模式）。
- `instructions` 可以是字符串/对象/数组；对象形态把问题放一个字段、数据放其余字段，用反引号路径（如 `` `ticket.messages[0].text` ``）引用 `state` 的嵌套字段。
- `state` 可以是字符串/JSON 对象/文本数组，仅限文本（无图像音视频）。
- 上下文限制：state+全部问题共 64k token；state+最长单问题 32k。

### 1.3 精确 HTTP 协议

```
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

请求体（官方示例原文）：

```json
{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "jev-latest",
  "questions": {
    "is_urgent": { "type": "noul", "instructions": "Does this convey urgency?",
                    "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"} },
    "department": { "type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"billing": "Payments, invoicing, refunds", "technical": "Bugs, outages, integrations", "sales": "Pricing, upgrades, new accounts"} },
    "frustration": { "type": "score", "instructions": "How frustrated is the customer?",
                     "criteria": ["Calm", "Frustrated", "Very angry"] }
  }
}
```

响应体（官方示例原文）：

```json
{
  "model": "jev-1.13.0",
  "answers": {
    "is_urgent": { "type": "noul", "noul": 0.95 },
    "department": { "type": "choice", "choice": "billing",
                    "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0},
                    "confidence": 0.81 },
    "frustration": { "type": "score", "score": 1.05,
                     "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                     "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
                     "confidence": 0.92 }
  },
  "usage": {"input_tokens": 304, "output_tokens": 18}
}
```

错误语义：`401`（key 无效）、`422`（请求体校验失败，body 指明出错字段）、`429`（限流：250k tokens/s、1200 req/min）、`529`（过载，退避重试）。模型别名 `jev-latest` → 当前 `jev-1.13.0`，响应 `model` 字段返回实际版本号。

SDK：Python `typesafe-sdk`（`TypeSafeClient().system_one(state=..., questions={...})`，`Choice/Score/Noul` 构造器）、JS `@typesafe-ai/sdk`（`choice()/score()/noul()` 函数）。answers 以问题 id 为键返回同名类型对象。

### 1.4 机器可读契约：官方 SDK 内嵌的 openapi.json（2026-09-22 补充）

官方 `typesafe-sdk` 包把官方 API 的 `openapi.json` 以 Pydantic schema 形式直接内置（`typesafe_sdk._schemas.models`，datamodel-codegen 生成）。这比文档正文更严格，平替对齐时纠正了若干假设：

- `GET /v1/models` 返回 `ModelMetadataList`：`{models: [{name, description, release_date}]}`——不是 OpenAI 风格的 id/object 列表。
- 422 响应体是 FastAPI 原生形状：`{"detail": [{"loc", "msg", "type"}]}`；其它错误用 `{"detail": "..."}`。
- `instructions` 在三种问题上**均可选**（`str | object | list | null`），与文档"必填"的说法不同。
- `NoulCriteria.true/false` 各自接受 null。
- score `criteria` 的 OpenAPI 下限是 1、无上限（文档正文"2 到 10 级"是建议而非强制）。Decidex 接受 1–26（26 = 字母读出容量），超出返回明确 422。
- ScoreAnswer 的 `legend` 原样回显 criteria 条目（含结构化值）。
- SDK 的响应包装层会把 legend/probabilities 的键转成 int，并提供 `.nouls`/`.choices`/`.scores` 分组访问器。

### 1.5 官方坦承的弱点（jaggedness 页面）

字面理解问题（否定/限定词按字面落地）；**不会数数**（逐项 Noul 替代）；**日期是文本不是有序量**（用枚举 Choice + 代码计算）；上下文腐烂（无关材料降低准确率）；state 不视为敌意输入（提示注入需自行防护）；不做任何生成（候选值用正则/LLM 拿到后让 Jev 挑选）。设计准则：*"避免问模型代码能精确计算的东西；避免把多个判断塞进一个问题。"*

## 2. 社区复刻版本

### 2.1 SemIf（原 OpenJev，TheoLeeCJ）—— 最重要的复刻

[github.com/TheoLeeCJ/SemIf](https://github.com/TheoLeeCJ/SemIf)（2.6k★，MIT）。自述：复刻的是**接口模式**（interface pattern），不是 Jev 的模型/训练。

核心方法 —— **"直接类型化 logits 读出"（direct typed logit readout）**：

```
state(非结构化) ─┐
criteria(运行时定义) ─┼→ 冻结的 4B 开源模型 ──原生选项 logits──→ 概率分布
typed options ─┘         （一次前向，不采样任何答案 token）
```

- 模型：Qwen3.5-4B（pin 了 revision；备选 MiniCPM5-2B、Qwen3-0.6B）。
- 选项以字母标注（A/B/C…），一次前向传播读取**末位对选项标识 token 的 logits**，softmax 成概率。无 JSON、无解码循环、无修复。
- **共享状态优化**：同一 state 服务多个判断时，prefill 一次（KV 前缀复用）+ 并行后缀。37 state × 21 criterion 实测：全新评分 2.33 决策/s → 前缀复用 10.75 → 并行后缀 **20.03**。
- 速度对比（同 4B 模型、21 个二元判断、RTX 3090）：直接 logits **1.023 s / 0 输出 token** vs 自回归生成 JSON 数组 5.332 s / 111 token（5.21×）。
- 质量（平衡准确率）：Qwen3-0.6B 0.440、MiniCPM5-2B 0.686、**Qwen3.5-4B 0.813**；对 TypeSafe 公开评测子集的一致率 0.845 vs 线上 Jev 0.883。原生 reranker 路线（Qwen3-Reranker）反而更差（0.625），直接 logits 是更优通用路线。

### 2.2 其它生态项目（模式参考）

均在一周内出现，共同模式：**循环/安全/算术留在普通代码，Jev 只做中间那个难以言状的窄判断**。

- `browser-use/jev-ultrafast`（641★）：浏览器 agent。页面→编号元素表，一次 Jev 调用同时选操作（CLICK/TYPE/…）和目标；TYPE 才调小 LLM。苏黎世→伦敦订票 7.1s / $0.0039。
- `awlevin/typesafe-computer-use`：OCR 屏幕符号化，Jev 选动作，$0.0002/步 vs Opus 5 的 $0.032/步。要点：*"前沿模型免费顺带做的每一步推理，在这里都必须重建成确定性状态。"*
- `jarrodwatts/jev-trader`：每 ~300ms 一个区块做一次买卖决策，模型延迟 ~81ms。
- `RomanSlack/jev-drone`：500Hz 控制律（代码）/ 50Hz 安全反射（代码）/ 15Hz 经典 CV / 2.5Hz Jev 仅作战术建议。
- `fhshaik/typesafe-mario`：模拟器 RAM→JSON 状态玩马里奥；`devagrawal09/jev-review` 分级代码评审；`AbdelStark/awesome-typesafe` 生态索引。
- `1kpapers.com`：1018 篇论文 24 主题分类——DeepSeek 总结花 $3.99，Jev 分类只花 $0.08（中位延迟 256ms）。

## 3. 对 Decidex 的设计决策

| 决策 | 依据 |
|---|---|
| 服务暴露**与官方逐字段一致**的 `POST /v1/systemone` + `GET /v1/models` | 官方 API 参考（§1.3）；做到官方 SDK/已有代码可换 base_url 直连 |
| `score = Σ p_i·i`、`confidence = clamp((K·p_max−1)/(K−1),0,1)`、probabilities 和为 1 | 官方示例验算 + 官方文档 JS 源码（§1.2） |
| 问题并行、彼此独立评估 | 官方语义（§1.2）；实现上每问题一次前向，互不共享上下文 |
| 默认引擎 = **LLM 直接 logits 读出**（Qwen3-4B 级模型 + 字母选项读出 + 温度校准） | SemIf 实测该路线质量最高、最接近 Jev 行为（§2.1） |
| 备选 embedding 引擎（sentence-transformers 相似度 + softmax） | 无 GPU/极低延迟场景；SemIf 的模型阶梯证明小模型也能给出可用概率 |
| Noul = 二选项（yes/no）读出 | Noul 无 confidence、本身即概率，与 Choice 共用读出机制 |
| `instructions`/`criteria` 支持 str|object|array，反引号路径引用 state | 官方结构化指令语义（§1.2） |
| 校验错误 422 + 指明字段、401/429 语义对齐 | 官方错误表（§1.3） |
| 与官方差异明确记录（RLCD 训练不可复刻，概率来自冻结 LLM 的 logits+温度） | SemIf 同样的立场声明（§2.1） |

## 4. 来源清单

- 官方博客：https://typesafe.ai/blog/introducing-system-one-models-and-jev
- 官方文档：https://docs.typesafe.ai/ （introduction / api / primitives / confidence / llms.txt / model-jaggedness）
- 官方 Python/JS SDK 文档：https://docs.typesafe.ai/sdk/python 、https://docs.typesafe.ai/sdk/javascript
- SemIf（原 OpenJev）：https://github.com/TheoLeeCJ/SemIf
- 实践指南（dev.to，Valyu）：https://dev.to/valyuai/how-to-use-jev-a-practical-guide-to-typesafes-system-one-model-g5e
- HN 讨论：https://news.ycombinator.com/item?id=49717558
- 生态索引：https://github.com/AbdelStark/awesome-typesafe
