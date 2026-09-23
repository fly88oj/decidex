# Jev 复刻项目全景对比（2026-09-23 调研）

> 数据源：GitHub / HuggingFace / Reddit (r/LLMDevs 287 项目综述) / apidog 复刻
> 综述 / Classmethod (森重) 对照表 / 项目官方页面，全部一手抓取。
> "复刻"定义：在本地重新实现 Jev 的**决策模型/接口**（区别于调用官方 API
> 的应用项目）。本文对比实现方式差异，并定位本项目（Decidex）的位置。

## 一、五条技术路线

所有复刻项目按实现机制分五类：

| 路线 | 机制 | 代表项目 |
|---|---|---|
| **A. 冻结模型字母/选项 logits 直读** | 完成式 prompt + 选项字母，一次前向读末位 logits，零生成 | SemIf、mini-jev、zhihz/openjev、jev-on-a-laptop、**Decidex（本项目）** |
| **B. 训练专用决策头** | 冻结大底座 + LoRA + 标量决策头（每候选一个标量分） | **Open-Jev (ZefanCai)**、AlexWortega/openjev |
| **C. 从零训练小评分器** | 选项 query 注意力 + 共享点积，字节嵌入或冻结编码器 | vinnylarouge/jevlike |
| **D. 并行受限解码** | KV 预填充一次 + 广播到每个字段 + 候选集受限 softmax | MLX 引擎（Apple Silicon）、vLLM PR #57250 |
| **E. 提示词包装** | 让任意 OpenAI 兼容模型按 Jev 格式输出 JSON | Sil/OpenJev（YouTube 推广的 wrapper） |

**共同盲区**：没有任何复刻实现 RLCD（官方的校准训练）。区别只在于概率
从哪来——A/D 是冻结 logits 的 softmax（"带差距的排序"），B/C 是训练目标
塑造的分布。

## 二、逐项目对比表

| 项目 | ★ | 底座 | 训练 | 读出机制 | 三原语 | 官方 API 兼容 | 官方一致率实测 | 特色 |
|---|---|---|---|---|---|---|---|---|
| **SemIf**（原 OpenJev, TheoLeeCJ） | 2.6k | Qwen3.5-4B 冻结 | 无 | 选项 token logits 直读 | 仅 choice | 无（自定义 JSONL/CLI） | 0.845（102 行官方子集，作者自测） | KV 前缀复用（10.75→20 决策/s）、WebGPU 浏览器版、模型阶梯评测 |
| **Open-Jev**（ZefanCai） | 热榜 | Qwen3.8-27B/9B/2B 冻结底座 | **LoRA+标量决策头**（15.5M 参数，148k 社区混合语料） | 专用决策头，每候选标量分 | choice+noul+score | **`/v1/systemone` 官方式** | JevBench 85.28%（Hard 72.07%） | 最完整复刻：训练头+温度保存+官方式服务+前缀缓存开关；4096 上限；需大显存 |
| **mini-jev**（r-ms） | 23 | Qwen3-4B-Instruct-2507 冻结 | 无 | 字母 logits | choice+noul（无 score） | 部分（本地 /run，自定义 schema） | 与语法约束 JSON 等精度（0.907 vs 0.909，6750 配对观察） | **预注册研究**（PREREG.md），量化"字母读出≈JSON 生成"；共享前缀缓存实测；诚实声明"非校准概率" |
| **vinnylarouge/jevlike** | 1.2k | 自训（字节嵌入或冻结 Qwen2.5-0.5B 编码器） | **有**（自备标签） | 选项注意力头+点积 | 仅 choice | 无 | 无官方对比（Wikispeedia 26% vs 8% 随机） | CPU 可跑；**视觉扩展**（Doom/国际象棋控制器，同一评分头） |
| **ekzhang/openjev-sglang** | — | Qwen3.6-35B-A3B NVFP4，SGLang/B200 | 无 | 每问 1 token 读出 | choice+noul | Jev 形 API | 无 | 生产级推理引擎路线，需 B200 |
| **daseinlabs/open-jev** | — | Gemma3-4B，MLX | 无 | 候选串整体评分 | choice | Jev 兼容 API | 无 | Apple Silicon 优化 |
| **jev-on-a-laptop**（rorshopping） | — | Qwen2.5-1.5B/7B、Qwen3-8B | 无 | 上下文读一次+逐项分数直读 | choice | 部分 | 无 | 笔记本友好 |
| **zhihz/openjev** | — | Qwen3-4B-Instruct-2507 | 无 | 每问读一次 next-token 分布 | choice | 无（中英接口） | 无 | 双语 |
| **MLX parallel-constrained-decoding** | HF Space | Qwen2.5-1.5B-4bit | 无 | KV 预填充广播+候选受限 softmax | 任意 schema 字段 | 无 | 无（仅延迟） | M4 Max 5.6–7× 加速、100% schema 合法 |
| **vLLM PR #57250** | 未合并 | DiffusionGemma | 无 | 扩散画布单槽位读取 | 三类 | OpenAI 兼容 | ~90%（语言分类） | 54 req/s@32 并发；有评审阻塞问题 |
| **AlexWortega/openjev** | — | Qwen3.5-4B | **有**（3 分类重训） | 蕴含式分类头 | choice | 无 | 无 | NLI 范式移植 |
| **Sil/OpenJev**（wrapper） | — | 任意 OpenAI 兼容 | 无 | 提示词+JSON 输出 | 三类 | 形似 | 无 | 零门槛但保留全部生成开销与解析风险（正是 Jev 要消灭的） |

## 三、本项目（Decidex）在全景中的位置

与上述项目的逐维对比：

| 维度 | Decidex | 最接近的同类 |
|---|---|---|
| 路线 | A（字母 logits 直读，SemIf 同源）+ **B 的蒸馏** | SemIf / mini-jev |
| 底座 | Qwen3-4B（延迟档）/ Qwen3-8B BF16/int4（一致率档） | Open-Jev 用 27B（更大但显存门槛高） |
| 训练 | **唯一以官方 API 实际输出为目标的蒸馏**（LoRA r32，17,954 样本，4 轮生成+挖矿） | Open-Jev 训练头但用社区混合语料（非官方答案）；vinnylarouge 自备标签 |
| 官方 API 兼容 | **逐字段 openapi 契约对齐 + 官方 Python/JS SDK 实测通过**（含 422 FastAPI 形状、别名解析） | 仅 Open-Jev 提供官方式端点；apidog 综述明确指出其余项目均不原生说官方 schema |
| 官方一致率 | **实测双端对比**：choice top-1 100%（JS 0.0025）、noul 决策 98.1%（v7）/100%（v8）、生成分布分歧率 26%→4.6% | SemIf 0.845（唯一其它有官方对比的）；其余无实测 |
| 性能工程 | KV 前缀复用（请求内+跨请求 LRU，长文档 11.4×）、OOM 降块、并发限流 | SemIf 有请求内前缀复用；MLX 引擎有广播式；本项目独有跨请求 LRU |
| 部署形态 | HF/PEFT 服务 + **GGUF Q4/Q6/Q8 + 蒸馏 0.6B MTP draft**（llama.cpp/Ollama/LM Studio 指南） | 无其它项目出 GGUF/MTP；SemIf 有 WebGPU 版 |
| 校准 | 温度拟合工具（留出集验证并诚实否决过拟合方案）+ 蒸馏隐式校准 | mini-jev 明确声明"非校准概率"；Open-Jev 保存训练温度 |

**本项目的三个独一无二**：
1. **蒸馏目标是官方真实输出**（经留出集验证闭环，而非合成标签）；
2. **契约级官方兼容**（openapi 逐字段 + 双官方 SDK 实测）；
3. **GGUF+MTP 生态导出**。

**他人的领先处**（如实记录）：Open-Jev 的专用决策头架构更接近"训练出
的决策模型"（我们是冻结读出+蒸馏）；SemIf 的 WebGPU 浏览器零安装体验；
MLX 引擎的字段级广播解码对结构化抽取更通用；vinnylarouge 的视觉决策
扩展（同一头打 Doom/国际象棋）是独特方向。

## 四、来源

- Reddit 287 项目综述：r/LLMDevs（chenrongwei）
- apidog 复刻综述：apidog.com/blog/openjev-open-source-jev-alternatives
- Classmethod 对照表：dev.classmethod.jp（森重，含 DGX Spark 实测）
- 项目一手页面：github.com/{TheoLeeCJ/SemIf, r-ms/mini-jev, vinnylarouge/jevlike,
  heyjunpenn/awesome-jev}、huggingface.co/ZefanCai/Open-Jev-27B-v1.1、
  zefan-cai.github.io/open-jev
