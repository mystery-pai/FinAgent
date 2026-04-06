# Eval 说明

本文档说明 Fin-Agent 当前的 RAG 评估方式、数据集格式、运行命令、指标含义和已知限制。

## 适用范围

当前评估入口是 [scripts/eval.py](/Users/adrian/LifeOS/Projects/AI-Coding/fin-agent/scripts/eval.py)，核心实现位于 [app/eval/ragas_evaluator.py](/Users/adrian/LifeOS/Projects/AI-Coding/fin-agent/app/eval/ragas_evaluator.py)。

这套评估分成两层：

- 检索评估：看目标文档是否被召回，以及排在第几位
- 生成评估：用 RAGAS 的 `faithfulness` 检查答案是否忠于检索上下文

## 快速开始

只评估检索效果，优先使用下面的命令：

```bash
python3 scripts/eval.py \
  --dataset tests/eval_questions.json \
  --mode hybrid \
  --metrics hit_rate mrr
```

如果要把结果保存为 JSON：

```bash
python3 scripts/eval.py \
  --dataset tests/eval_questions.json \
  --mode hybrid \
  --metrics hit_rate mrr \
  --output data/eval_results/latest.json
```

如果要同时评估 `faithfulness`：

```bash
python3 scripts/eval.py \
  --dataset tests/eval_questions.json \
  --mode hybrid \
  --metrics hit_rate mrr faithfulness
```

## 参数说明

`scripts/eval.py` 支持以下参数：

- `--dataset`: 评测数据集路径，默认 `tests/eval_questions.json`
- `--mode`: 检索模式，可选 `bm25`、`vector`、`hybrid`
- `--top-k`: 检索深度，默认 `5`
- `--metrics`: 评估指标，可选 `hit_rate`、`mrr`、`faithfulness`
- `--output`: 可选，保存结果到 JSON 文件

## 数据集格式

当前评测集是一个 JSON 数组，每个元素对应一条样本。

示例：

```json
{
  "question": "2025 财年大中华区（Greater China）的销售表现如何？主要受哪些产品影响？",
  "filters": {
    "year": 2025
  },
  "expected_doc_ids": ["2025_11"],
  "reference_answer": "2025 财年大中华区的净销售额为 643.77 亿美元，较 2024 年下降了 4%。这一趋势主要是由于 iPhone 的销售额下降所致，但部分被 Mac 销售额的增长所抵消。数据来源: Item 7 - Management’s Discussion and Analysis (Segment Operating Performance)."
}
```

字段说明：

- `question`: 用户问题
- `filters`: 检索过滤条件，目前常用 `year`
- `expected_doc_ids`: 期望命中的文档 ID
- `expected_chunk_ids`: 可选，期望命中的 chunk ID，格式如 `2025_11_3`
- `reference_answer`: 参考答案，用于人工比对，也会写入评估结果

## 指标定义

### `hit_rate`

含义：目标文档是否出现在前 `top-k` 结果中。

计算方式：

- 命中则为 `1.0`
- 未命中则为 `0.0`

### `mrr`

含义：第一个相关结果的倒数，越靠前越好。

计算方式：

- 第 1 位命中，`MRR = 1.0`
- 第 2 位命中，`MRR = 0.5`
- 第 3 位命中，`MRR = 0.3333`
- 未命中，`MRR = 0.0`

### `faithfulness`

含义：生成答案是否忠于检索上下文。

当前实现通过 RAGAS 调用 DeepSeek 模型完成，不是纯本地评估。

## 输出结果说明

评估结果包含两部分：

- `summary`: 汇总指标
- `samples`: 每条样本的详细结果

单条样本结果里，最有用的字段通常是：

- `retrieval_query`: 实际用于检索的查询
- `query_type`: 问题类型，如 `factual`、`summary`
- `first_relevant_rank`: 第一个相关结果的排名
- `retrieved_doc_ids`: 实际召回的文档 ID 列表
- `retrieval_debug`: 查询解析和过滤信息

## 推荐使用方式

### 场景一：先看检索是否可靠

优先只跑：

```bash
python3 scripts/eval.py --metrics hit_rate mrr
```

原因很直接：

- `hit_rate` 和 `mrr` 足够暴露大部分召回和排序问题
- 不依赖远程 LLM，排障更稳定
- 更适合调整 `expected_doc_ids`、`top-k` 和检索规则

### 场景二：排查单条样本为什么失败

配合 [scripts/debug_retrieve.py](/Users/adrian/LifeOS/Projects/AI-Coding/fin-agent/scripts/debug_retrieve.py)：

```bash
python3 scripts/debug_retrieve.py \
  "Summarize Apple's Item 1 business overview in 2025" \
  --mode hybrid
```

先回答两个问题：

- 是完全没召回目标文档
- 还是目标文档召回了，但排序太靠后

这两个问题不能混为一谈。

### 场景三：再看生成是否忠于证据

只有在检索结果已经基本稳定后，再启用 `faithfulness`。否则你会把检索问题和生成问题混在一起，结论会失真。

## 当前实现细节

### 检索评估逻辑

[app/eval/ragas_evaluator.py](/Users/adrian/LifeOS/Projects/AI-Coding/fin-agent/app/eval/ragas_evaluator.py) 的 `_first_relevant_rank` 逻辑是：

- 如果召回结果中的 `doc_id` 命中 `expected_doc_ids`
- 或者 `chunk_id` 命中 `expected_chunk_ids`
- 则认为该条结果相关

这意味着当前检索评估本质上是“目标文档命中评估”，不是开放式语义人工打分。

### 生成评估逻辑

当 `--metrics` 中包含 `faithfulness` 时：

- 评估器会初始化 `AnswerGenerator`
- 先生成答案
- 再把 `question`、`response`、`retrieved_contexts` 送给 RAGAS

如果不包含 `faithfulness`，评估器会构造一个不依赖远程 LLM 的 `AnswerGenerator`，用于降低运行门槛。

## 已知限制

### 1. `faithfulness` 依赖 DeepSeek

当前实现写死要求：

- `settings.llm_provider == "deepseek"`
- 存在有效的 `DEEPSEEK_API_KEY`

否则会直接报错。

### 2. 连接失败时 `faithfulness` 结果可能不可用

如果答案生成或 RAGAS 调用过程中出现连接错误，样本级别的 `faithfulness` 可能会变成 `nan`，汇总值会因为忽略 `nan` 而退化为 `0.0`。

这时：

- `hit_rate` / `mrr` 仍然有参考价值
- `faithfulness` 没有参考价值

### 3. 评测问题必须和标准来源一致

如果问题问得太宽，而 `expected_doc_ids` 又写得太窄，就会出现“系统答得不算错，但评测判错”的情况。

例如：

- 问题写成宽泛的 `business overview`
- 标准却只允许命中 `Item 1 Business`

这类情况应该先修正评测口径，再讨论检索效果。

### 4. 当前更偏向文档命中评估，不是最终业务质量评估

当前评测更适合衡量：

- 是否召回正确 section
- 排序是否合理

不适合直接代表：

- 用户是否真正满意
- 答案是否完整
- 答案中文表达是否自然

## 当前评测集位置

默认评测集路径：

- [tests/eval_questions.json](/Users/adrian/LifeOS/Projects/AI-Coding/fin-agent/tests/eval_questions.json)

如果新增样本，建议遵循两个原则：

- 问题描述要明确，避免标准来源歧义
- `expected_doc_ids` 尽量指向真正的主证据文档，不要为了“跑绿”而随意放宽

## 建议的评估顺序

推荐按这个顺序执行：

1. 先补齐或修正 `tests/eval_questions.json`
2. 运行 `hit_rate + mrr`
3. 用 `debug_retrieve.py` 分析失败样本
4. 调整检索逻辑或评测标准
5. 检索稳定后，再运行 `faithfulness`

这条路径比一开始就追 `ragas` 总分更短，也更可靠。
