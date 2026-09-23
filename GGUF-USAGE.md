# GGUF 版使用指南（llama-server / Ollama / LM Studio）

> 适用：`gguf/decidex-core-8b-r1-Q4_K_M / Q6_K / Q8_0.gguf`（v7 蒸馏版，已合并进
> Qwen3-8B）与 `decidex-draft-0.6b-Q8_0.gguf`（MTP draft）。
> **核心概念：这些是决策引擎模型，不是聊天模型。** 用完成式 prompt + 贪心
> 解码，读首字母（argmax）或读 logprobs（概率分布）。实测：经 llama-server
> 读出的 noul=0.9503 / score=1.0404，与官方文档示例值（0.95 / 1.05）贴合。

## 万物共用的决策 Prompt 模板

```
You are a precise decision engine. Evaluate the STATE against the
QUESTION and pick the single best option.

STATE:
{state}

QUESTION:
{instruction}

OPTIONS:
A. {选项1描述}
B. {选项2描述}
C. {选项3描述}

Answer with the letter of the single best option.
Answer:
```

三种原语都是这个模板：**Noul** = 两个选项 `A. yes / B. no`；**Choice** =
每个选项一行；**Score** = 每个等级一行（等级下标即得分）。答案就是
`Answer:` 后的第一个 token（字母 A/B/C…）。

三个关键设置（所有平台一致）：
- **temperature = 0**（贪心）
- **max_tokens / n_predict = 1–4**（答案只有字母）
- **不要套 chat 模板**——Qwen3-8B 是 thinking 混合模型，聊天模板会插入
  思考过程破坏字母读出。一律用 raw completion 接口。

---

## 1. llama-server（推荐——唯一能读概率分布的路径）

```bash
# 基本启动（GPU）
llama-server -m gguf/decidex-core-8b-r1-Q4_K_M.gguf -c 8192 --port 8080 -ngl 99

# 带 MTP（draft 投机解码；注意 MTP 加速"生成"，对单字母概率读出无收益，
# 适合把这个模型当普通 LLM 生成文本时用）
llama-server -m gguf/decidex-core-8b-r1-Q4_K_M.gguf \
  --model-draft gguf/decidex-draft-0.6b-Q8_0.gguf \
  --spec-draft-n-max 4 -ngl 99 -c 8192
```

### 读概率分布（= Jev 决策 API）

`POST /completion` 带 `n_probs`，取返回里 `completion_probabilities[0]`
的 `top_logprobs`，把字母 token 的 `exp(logprob)` 聚合归一即可：

```bash
curl -s http://127.0.0.1:8080/completion -H "Content-Type: application/json" -d '{
  "prompt": "<上面的模板填好后>",
  "n_predict": 1, "temperature": 0, "n_probs": 20
}'
```

现成客户端（实现了 noul/choice/score 三原语 + 官方 confidence 公式，
开箱即用）：

```bash
python examples/gguf_decision_client.py --base-url http://127.0.0.1:8080
# noul  : 0.9503
# choice: {"choice": "billing", "probabilities": {"billing": 0.9989, ...}, "confidence": 0.9984}
# score : {"score": 1.0404, "legend": {...}, "probabilities": {...}, "confidence": 0.9349}
```

只想要 argmax：`POST /completion` 不带 `n_probs`，返回的 `content`
首字符就是字母。

---

## 2. Ollama

```dockerfile
# Modelfile（与 gguf 同目录）
FROM ./decidex-core-8b-r1-Q4_K_M.gguf
```

```bash
ollama create decidex-core -f Modelfile
```

用 raw 模式（跳过模板包装）调 `/api/generate`：

```bash
curl -s http://127.0.0.1:11434/api/generate -d '{
  "model": "decidex-core",
  "prompt": "<决策模板填好>",
  "raw": true,
  "stream": false,
  "options": {"temperature": 0, "num_predict": 2}
}'
# 返回的 response 字段首字符即答案字母
```

或命令行：`ollama run decidex-core --verbose` 后粘贴模板（交互式）。

**限制（诚实说明）**：Ollama 的 API 不暴露 logprobs，所以走 Ollama 只能拿
argmax 字母，拿不到概率分布/confidence。需要分布请用 llama-server。

---

## 3. LM Studio

1. 把 `.gguf` 放入模型目录（或在 GUI 里 Import），模型名会带
   `decidex-core-8b` 字样；
2. 开启本地 Server（默认 `http://127.0.0.1:1234`，OpenAI 兼容）；
3. 用 **completions** 端点（不是 chat）：

```bash
curl -s http://127.0.0.1:1234/v1/completions -H "Content-Type: application/json" -d '{
  "model": "decidex-core-8b-r1-Q4_K_M",
  "prompt": "<决策模板填好>",
  "temperature": 0, "max_tokens": 2
}'
# choices[0].text 首字符 = 答案字母
```

LM Studio 的 completions 支持 `logprobs`/`top_logprobs` 参数时同样可以
读字母分布（与 llama-server 同法聚合）；是否可用取决于其版本。

GUI 直接聊天页也可以粘贴模板测试——但注意它默认套聊天模板，Qwen3 可能
先输出思考标签；在模型设置里关掉 thinking / system prompt 后再试。

---

## 常见问题

- **输出了一堆思考文字而不是字母** → 平台套了 chat 模板。改用 raw
  completion（llama-server `/completion`、Ollama `raw:true`、LM Studio
  `/v1/completions`），或在 chat 里显式关闭 thinking。
- **要概率分布** → 只有暴露 logprobs 的路径可行（llama-server `n_probs`
  最稳）。
- **MTP/draft 有用吗** → 只对多 token 生成有用（当普通 LLM 用时）；
  决策读出只生成 1 个 token，投机解码无从加速。
- **和 Decidex 本地服务什么关系** → Decidex 的 `python -m decidex serve`
  走 HF/PEFT 路径（前缀复用、批量、官方 SDK 兼容 API），功能最全；
  GGUF 路线面向 llama.cpp 生态的轻量部署。
