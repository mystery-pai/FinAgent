# Fin-Agent RAG 系统设计文档

**项目**: AAPL 10-K 智能问答系统
**设计日期**: 2025-03-28
**设计目标**: 面试友好的 RAG 系统，重点展示设计能力、工程完整性和取舍意识

---

## 1. 整体架构

### 系统分层设计

系统采用 5 层架构，每层职责单一：

**1. Ingest Layer（数据摄取层）**
- 读取 `aapl_10k.json`，将原始 SQL 导出格式转换为标准化文档结构
- 文档主键：`{year}_{section_id}`
- 添加归一化的 `item_type` 字段（如 "risk_factors", "md&a"）

**2. Index Layer（索引层）**
- **BM25 索引**：使用 `bm25s` 库，负责关键词精确召回（年份、会计科目、专有名词）
- **向量索引**：使用 ChromaDB，存储 `bge-small-en-v1.5` embeddings，负责语义召回

**3. Retrieve Layer（检索层）**
- **Query Parser**：提取年份约束（"2025"、"近三年"）和主题（风险、营收、供应链）
- **Hybrid Retriever**：并行执行 BM25 + 向量检索，使用 RRF（Reciprocal Rank Fusion）合并结果
- **Reranker**：基于规则的加权（问风险优先 Item 1A，问财务优先 Item 8）

**4. Generate Layer（生成层）**
- 接收 top-k chunks，使用本地 LLM 生成答案
- Prompt 约束：输出结论 + 证据引用，禁止编造
- 支持中英文问答

**5. Serve Layer（服务层）**
- **Streamlit UI**：简单的问答界面，显示答案、引用、检索调试信息
- **FastAPI**：可选的 REST API，便于程序化调用

### 技术栈

| 组件 | 技术选型 | 理由 |
|------|----------|------|
| 向量存储 | ChromaDB | 轻量、本地运行、无需独立服务 |
| Embedding | bge-small-en-v1.5 | 英文金融文本表现好，384维，适合本地部署 |
| 关键词检索 | bm25s | 纯Python库，无额外依赖 |
| 结果融合 | RRF | 简单有效的排序列表融合算法 |
| LLM | 本地小模型 (7B级别) | 可替换，重点是检索层 |
| 前端 | Streamlit | 快速构建交互界面，面试演示友好 |
| 容器化 | Docker + docker-compose | 一键运行 |

---

## 2. 数据模型与分块策略

### 标准化文档结构

**输入数据格式**（现有 aapl_10k.json）：
```json
{
  "symbol": "AAPL",
  "file_fiscal_year": 2025,
  "form_type": "10-K",
  "section_title": "Item 1A. Risk Factors",
  "section_id": 3,
  "section_text": "..."
}
```

**转换后的标准化格式**：
```python
{
  "doc_id": "2025_3",              # {year}_{section_id}
  "symbol": "AAPL",
  "year": 2025,
  "form_type": "10-K",
  "section_id": 3,
  "section_title": "Item 1A. Risk Factors",
  "item_type": "risk_factors",     # 归一化类型
  "text": "原始section文本",
  "metadata": {
    "fiscal_year": 2025,
    "form_type": "10-K"
  }
}
```

### Section-Aware 分块策略

**原则**：不按固定长度粗暴切分，而是：
1. **以 section 为一级文档单位**（保持上下文完整性）
2. 对超长 section（>1000 tokens）做二级 chunking
3. chunk 大小：500-1000 tokens，overlap 10%
4. **保留表格边界**：Item 8 的财务表格不与叙述段混在一起

**Chunk 元数据**：
```python
{
  "chunk_id": "2025_3_0",          # {year}_{section_id}_{chunk_order}
  "doc_id": "2025_3",
  "year": 2025,
  "section_title": "Item 1A. Risk Factors",
  "item_type": "risk_factors",
  "chunk_order": 0,
  "text": "chunk内容..."
}
```

### Item 类型归一化映射

```python
ITEM_TYPE_MAP = {
  "Item 1": "business",
  "Item 1A": "risk_factors",
  "Item 7": "md&a",
  "Item 8": "financial_statements",
  "Item 3": "legal",
  "Item 11": "executive_compensation",
  # 其他归为 "other"
}
```

---

## 3. 检索系统设计

### 混合检索架构

**为什么需要混合检索？**
- **BM25**：擅长精确匹配（"2025年 Net sales"、"Item 1A"、专有名词）
- **Vector**：擅长语义理解（"供应链挑战"、"竞争压力变化"）
- 金融问答需要两者结合才能保证准确性和召回率

### 检索流程

```
用户问题: "Apple 2025年的主要风险有哪些?"
    ↓
【Step 1: Query Understanding】
- 提取年份: 2025
- 提取主题: "风险" → item_type = "risk_factors"
- 问题类型: factual (单年问答)
    ↓
【Step 2: 元数据预过滤】
year = 2025 AND item_type = "risk_factors"
    ↓
【Step 3: 并行检索】
├─ BM25 Retriever → top-10 chunks (关键词匹配)
└─ Chroma Retriever → top-10 chunks (语义相似度)
    ↓
【Step 4: RRF 融合】
合并去重 → 按融合分数重排序
    ↓
【Step 5: 规则 Rerank】
- 匹配到 "Item 1A" 的 chunk 加权
- 年份精确匹配的 chunk 加权
    ↓
【Step 6: Top-k 输出】
返回 top-5 chunks 给生成层
```

### RRF (Reciprocal Rank Fusion) 算法

```python
def rrf_merge(bm25_results, vector_results, k=60):
    """
    融合两个排序列表
    k 是平滑参数，通常取 60
    """
    scores = {}

    for rank, doc in enumerate(bm25_results):
        doc_id = doc['chunk_id']
        scores[doc_id] = scores.get(doc_id, 0) + 1/(k + rank + 1)

    for rank, doc in enumerate(vector_results):
        doc_id = doc['chunk_id']
        scores[doc_id] = scores.get(doc_id, 0) + 1/(k + rank + 1)

    # 按融合分数降序排列
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### 规则 Rerank 策略

```python
RERANK_RULES = {
    "risk_keywords": ["risk", "uncertainty", "challenge"],
    "priority_item": "Item 1A",  # 问风险时优先
    "year_bonus": 20,            # 年份精确匹配加分
    "item_bonus": 15             # item类型匹配加分
}
```

---

## 4. 问答生成与输出格式

### Prompt 模板设计

```python
SYSTEM_PROMPT = """You are a senior financial analyst specializing in 10-K reports.
Your task is to answer questions based ONLY on the provided context chunks.

Rules:
1. Answer based STRICTLY on the given context
2. If context is insufficient, say "Insufficient information to answer"
3. Always cite the source (year, section_title)
4. For numerical questions, be precise
5. For comparison questions, compare by year"""

USER_PROMPT_TEMPLATE = """
Question: {query}

Context:
{context_chunks}

Answer in this format:

## 结论
[Your answer here]

## 依据
- {year}, {section_title}: [brief evidence]
- {year}, {section_title}: [brief evidence]

## 补充说明
[Additional context if needed, otherwise omit]
"""
```

### API 响应格式

```json
{
  "answer": "根据2025年10-K报告，Apple的主要风险包括...",
  "citations": [
    {
      "year": 2025,
      "section_title": "Item 1A. Risk Factors",
      "chunk_id": "2025_3_2",
      "snippet": "The Company's business is subject to..."
    }
  ],
  "retrieval_debug": {
    "query_type": "factual",
    "year_filter": [2025],
    "item_type_filter": "risk_factors",
    "bm25_count": 10,
    "vector_count": 10,
    "final_top_k": 5
  }
}
```

### 支持的三类问题

1. **单点事实问答**
   - 例："Apple 2025年的净销售额是多少？"
   - 策略：单年检索 + 精确数值提取

2. **跨年变化分析**
   - 例："2023到2025年Apple在风险披露上有哪些变化？"
   - 策略：多年检索 + 对比生成

3. **简短报告生成**
   - 例："总结Apple近三年经营表现"
   - 策略：Item 7 + Item 8 跨年检索 + 结构化摘要

---

## 5. 项目结构

```
fin-agent/
├── app/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # FastAPI路由
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # 配置管理
│   │   └── models.py          # Pydantic模型定义
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── parser.py          # JSON解析
│   │   └── chunker.py         # 文本分块
│   ├── retrieve/
│   │   ├── __init__.py
│   │   ├── bm25_retriever.py  # BM25检索
│   │   ├── chroma_retriever.py # Chroma向量检索
│   │   ├── hybrid_retriever.py # 混合检索融合
│   │   └── query_parser.py    # 查询解析
│   ├── generate/
│   │   ├── __init__.py
│   │   ├── prompter.py        # Prompt模板
│   │   └── generator.py       # 答案生成
│   └── schemas/
│       ├── __init__.py
│       └── document.py        # 文档schema定义
├── data/
│   ├── raw/
│   │   └── aapl_10k.json      # 原始数据
│   └── processed/
│       └── chunks/            # 处理后的chunk
├── ui/
│   └── streamlit_app.py       # Streamlit界面
├── scripts/
│   ├── build_index.py         # 构建索引脚本
│   └── eval.py                # 评估脚本
├── tests/
│   ├── test_retriever.py
│   └── test_e2e.py
├── embeddings/                # 本地embedding模型缓存
├── indexes/                   # BM25和Chroma索引
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── README.md
└── project.md
```

---

## 6. 实现优先级

基于面试场景，建议按以下顺序实现：

1. ✅ 数据标准化 - 清洗JSON，统一格式
2. ✅ Section/Chunk建模 - 实现智能分块
3. ✅ BM25检索 - 关键词精确匹配
4. ✅ Embedding检索 - Chroma + bge-small
5. ✅ 混合检索 - RRF融合
6. ✅ 问答接口 - 基础QA API
7. ✅ 引用输出 - 可追溯答案
8. ✅ Streamlit UI - 简单问答界面
9. ✅ Docker化 - 一键部署
10. ✅ README - 完整文档
11. ✅ 评估脚本 - 质量验证

---

## 7. 设计亮点（面试重点）

1. **Section-Aware Chunking**
   - 为什么不纯定长切分：保持10-K结构完整性
   - 表格边界处理：Item 8财务表不混入叙述段

2. **混合检索设计**
   - 为什么用 BM25 + Vector：金融术语精确性 + 语义理解
   - RRF vs 加权融合：模型无关，简单有效

3. **元数据过滤**
   - year 和 item_type 预过滤：减少检索空间，提升精度
   - 优先级规则：问风险优先 Item 1A，问财务优先 Item 8

4. **可追溯引用**
   - 每个答案带年份 + section_title 引用
   - retrieval_debug 展示检索过程

5. **本地化部署**
   - 无外部依赖：Chroma 本地，Embedding 本地
   - Docker一键运行：简化演示流程

---

## 8. 待讨论事项

- [ ] LLM 最终选型（Qwen2.5-7B vs Llama-3.1-8B vs 其他）
- [ ] 是否实现 Reranker（bge-reranker-base vs 规则）
- [ ] 中文问题处理策略（翻译 vs multilingual embedding）
- [ ] 评估集规模（10-20条手工问题 vs 自动生成）

---

**下一步**: 准备实现阶段
