这份方案旨在帮助你构建一个基于 **RAG (Retrieval-Augmented Generation)** 架构的智能金融问答系统。由于要求使用本地开源模型并提供容器化环境，我们将采用 **LangChain/LlamaIndex + ChromaDB + Ollama (或 HuggingFace Transformers)** 的技术栈。

---

### 项目方案名称：Apple-Insight 10-K 智能财报分析系统

---

### 1. 核心技术选型
*   **语言**: Python 3.10+
*   **前端 UI**: Streamlit (快速构建交互式金融看板)
*   **向量数据库**: ChromaDB (轻量级、本地运行)
*   **嵌入模型 (Embedding)**: `BAAI/bge-small-en-v1.5` (开源、金融文本表现优异、轻量)
*   **大语言模型 (LLM)**: `Llama-3.1-8B-Instruct` 或 `Qwen2.5-7B-Instruct` (本地运行，建议通过 Ollama 暴露 API)
*   **RAG 框架**: LlamaIndex (在处理结构化文档和金融表格方面比 LangChain 更具针对性)

---

工程结构
fin-agent/
├── app/
│   ├── api/
│   ├── core/
│   ├── ingest/
│   ├── index/
│   ├── retrieve/
│   ├── generate/
│   └── schemas/
├── data/
│   ├── raw/
│   └── processed/
├── scripts/
│   ├── build_index.py
│   └── eval_sample.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── README.md
└── project.md


### 2. 核心模块设计

#### 模块 A：数据预处理与结构化分块 (Ingestion)
10-K 文件的难点在于**表格与文本混排**。
*   **分层索引策略**: 按照 `file_fiscal_year` -> `section_title` -> `chunk` 进行索引。
*   **文本清洗**: 去除无效的转义字符，识别并保留 Item 8 中的数值表格（建议将 ASCII 表格转换为 Markdown 格式以增强 LLM 理解力）。
*   **语义分块**: 使用 `RecursiveCharacterTextSplitter`，块大小建议 512-1024 token，重叠 10%。

#### 模块 B：混合检索系统 (Hybrid Search)
金融问答需要极高的准确性，单纯的语义搜索（Vector）可能找不到精确的财务数值。
*   **向量搜索**: 处理“公司的竞争风险有哪些？”等语义问题。
*   **关键字搜索 (BM25)**: 处理“2025年的 Net sales 是多少？”等包含特定专有名词的问题。
*   **重排序 (Rerank)**: 使用 `bge-reranker-base` 对检索到的前 10 个片段进行精排，选出前 3 个。

#### 模块 C：金融增强生成 (Generation)
*   **Prompt Engineering**: 设定系统角色为“资深金融分析师”。强制要求模型：*“如果检索到的内容中没有数据，请回答不知道，不要编造数字”*。
*   **多维比较查询**: 实现一个专门的逻辑来处理跨年度比较（例如：对比 2024 和 2025 的营收增长）。

---

### 3. README.md 结构建议

一份专业的 README 是加分项，应包含：
1.  **项目简介**: 解决 10-K 报告长、难、数值密集的痛点。
2.  **快速启动指南**: 
    *   `docker-compose up -d`
    *   访问 `localhost:8501`。
3.  **技术亮点**:
    *   **跨年度对比能力**: 如何通过 Metadata Filter 实现指定年份的查询。
    *   **表格感知检索**: 针对 Item 8 的处理方案。
4.  **评估指南 (Eval)**:
    *   如何评估回答的准确性（如使用 RAGAS 框架计算忠实度 Faithfulness）。
5.  **AI 协作说明**: 说明哪些代码模块是辅助生成的，你如何进行重构和逻辑校验。

---

### 4. 容器化配置 (docker-compose.yml 示例)

```yaml
services:
  # LLM 推理引擎
  ollama:
    image: ollama/ollama
    volumes:
      - ./ollama_data:/root/.ollama
    ports:
      - "11434:11434"

  # 主应用
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
    depends_on:
      - ollama
    volumes:
      - ./data:/app/data
```

---

### 5. 实现阶段拆解 (Git 提交建议)

为了展示良好的开发习惯，建议 Git 提交历史如下：
1.  `feat: project initialized and docker config`
2.  `feat: parser for aapl_10k.json to extract text and tables`
3.  `feat: implement vector storage with ChromaDB and BGE embeddings`
4.  `feat: develop hybrid retrieval logic (BM25 + Vector)`
5.  `feat: add financial analysis prompt template and LLM chain`
6.  `feat: streamlit dashboard for Q&A and visualization`
7.  `refactor: optimize chunking strategy for Item 8 tables`
8.  `docs: complete README and evaluation guide`

---

### 6. 针对样例数据的特别处理建议

观察你提供的数据：
*   **Item 8 包含大量数值**: 如 `Net income 112,010`。在预处理时，将这些 JSON 里的表格行提取出来，并打上 `year: 2025`, `type: income_statement` 的标签（Metadata）。
*   **跨年问答**: 如果用户问“Apple近三年的净利润趋势”，你的系统应能触发 `Metadata-driven retrieval`，搜集 2023-2025 的 Item 8 数据，交由 LLM 总结。

### 7. 避坑指南
*   **内存管理**: 本地运行 Llama 3.1 需要至少 8GB 显存或 16GB 内存。如果在 CPU 运行，务必使用 **Quantized (量化版)** 模型（如 GGUF 格式）。
*   **解析效率**: 不要一次性把所有文本塞给 LLM，10-K 片段很长， context window 溢出会导致报错或遗忘。
