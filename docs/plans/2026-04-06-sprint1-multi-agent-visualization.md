# Sprint 1: Multi-Agent 架构 + 财报可视化基础实现

**创建日期**: 2026-04-06
**负责人**: Pike
**预计工期**: 3-4 天
**状态**: 📋 Planning

---

## 🎯 Sprint 目标

构建 fin-agent 的 multi-agent 架构基础和最小可用的财报数据可视化功能（MVP）。

### 核心交付物

1. ✅ **Multi-Agent 基础架构**
   - Router Agent（意图识别）
   - Agent Orchestrator（统一编排）
   - 可扩展的 Agent 基类

2. ✅ **数据提取能力**
   - LLM Function Calling 实现
   - 结构化数据提取（从文本到图表数据）
   - 数据验证和错误处理

3. ✅ **可视化引擎**
   - Plotly 图表生成器
   - 支持折线图（趋势分析）
   - 交互式图表输出

4. ✅ **API 端点**
   - `/visualize` 新端点
   - 请求/响应模型定义
   - 基础错误处理

5. ✅ **单元测试**
   - 数据提取测试
   - 图表生成测试
   - 端到端集成测试

---

## 📐 技术架构设计

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │          AgentOrchestrator (统一入口)                   │ │
│  │                                                          │ │
│  │  ┌──────────────┐                                       │ │
│  │  │ RouterAgent  │ ─── 意图识别 ───┐                    │ │
│  │  └──────────────┘                  │                    │ │
│  │                                     ↓                    │ │
│  │  ┌──────────────┐         ┌───────────────┐            │ │
│  │  │   QAAgent    │ ←───────│  Route Logic  │            │ │
│  │  └──────────────┘         └───────────────┘            │ │
│  │         │                         │                     │ │
│  │         │                         ↓                     │ │
│  │  ┌──────────────────────────────────────┐              │ │
│  │  │      VisualizationAgent              │              │ │
│  │  │                                      │              │ │
│  │  │  ┌────────────────┐                 │              │ │
│  │  │  │ DataExtractor  │ (LLM Function)  │              │ │
│  │  │  └────────────────┘                 │              │ │
│  │  │           ↓                          │              │ │
│  │  │  ┌────────────────┐                 │              │ │
│  │  │  │ChartGenerator  │ (Plotly)        │              │ │
│  │  │  └────────────────┘                 │              │ │
│  │  └──────────────────────────────────────┘              │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         Existing Components (复用)                      │ │
│  │  - HybridRetriever (BM25 + Chroma)                     │ │
│  │  - AnswerGenerator (LLM Client)                        │ │
│  │  - ConversationManager (Session)                       │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 新增模块结构

```
fin-agent/
├── app/
│   ├── agents/                          # 🆕 Multi-Agent 核心模块
│   │   ├── __init__.py
│   │   ├── base.py                      # Agent 抽象基类
│   │   ├── orchestrator.py              # Agent 编排器（核心路由）
│   │   ├── router.py                    # Intent 识别 Agent
│   │   ├── qa_agent.py                  # QA Agent 封装
│   │   └── visualization_agent.py       # 可视化 Agent
│   │
│   ├── tools/                           # 🆕 Agent 工具集
│   │   ├── __init__.py
│   │   ├── data_extractor.py            # LLM 数据提取工具
│   │   └── chart_generator.py           # Plotly 图表生成器
│   │
│   ├── schemas/
│   │   └── models.py                    # 🔄 新增数据模型
│   │       └── + VisualizationRequest
│   │       └── + VisualizationResponse
│   │       └── + ChartDataSchema
│   │       └── + UnifiedResponse
│   │
│   ├── api/
│   │   └── main.py                      # 🔄 新增 API 端点
│   │       └── POST /visualize
│   │
│   └── core/
│       └── config.py                    # 🔄 新增配置项
│           └── + visualization_config
│
├── tests/                               # 🆕 单元测试
│   ├── test_agents/
│   │   ├── test_router.py
│   │   └── test_visualization_agent.py
│   └── test_tools/
│       ├── test_data_extractor.py
│       └── test_chart_generator.py
│
└── requirements.txt                     # 🔄 新增依赖
    └── + plotly>=5.18.0
    └── + pydantic>=2.0.0
```

---

## 📋 详细任务分解

### Task 1: 创建 Agent 基础架构 (4-5h)

#### 1.1 创建模块结构 (30min)

```bash
# Create directories
mkdir -p app/agents app/tools tests/test_agents tests/test_tools

# Create __init__.py files
touch app/agents/__init__.py
touch app/tools/__init__.py
touch tests/test_agents/__init__.py
touch tests/test_tools/__init__.py
```

#### 1.2 实现 Agent 基类 (1h)

**文件**: `app/agents/base.py`

```python
# Abstract base class for all agents
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel


class AgentConfig(BaseModel):
    """Agent configuration"""
    name: str
    description: str
    version: str = "1.0.0"
    enabled: bool = True


class AgentResult(BaseModel):
    """Unified agent result format"""
    agent_name: str
    success: bool
    data: Dict[str, Any]
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class BaseAgent(ABC):
    """
    Abstract base class for all agents

    All agents must implement:
    - execute(): Main execution logic
    - validate_input(): Input validation
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name

    @abstractmethod
    async def execute(self, **kwargs) -> AgentResult:
        """
        Execute agent logic

        Args:
            **kwargs: Agent-specific parameters

        Returns:
            AgentResult with execution results
        """
        pass

    @abstractmethod
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters

        Returns:
            True if valid, raises ValueError otherwise
        """
        pass

    def _create_result(
        self,
        success: bool,
        data: Dict[str, Any],
        error: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> AgentResult:
        """Helper to create AgentResult"""
        return AgentResult(
            agent_name=self.name,
            success=success,
            data=data,
            error=error,
            metadata=metadata or {}
        )
```

**关键设计点**:
- 使用 ABC 强制子类实现核心方法
- 统一的 AgentResult 格式便于编排
- 配置驱动，易于扩展

#### 1.3 实现 RouterAgent (1.5h)

**文件**: `app/agents/router.py`

```python
# Intent classification agent
from typing import Literal, Optional
from pydantic import BaseModel, Field
import re

from app.agents.base import BaseAgent, AgentConfig, AgentResult
from app.generate.answer_generator import AnswerGenerator


class IntentClassification(BaseModel):
    """Intent classification result"""
    intent: Literal["qa", "visualization", "report"] = Field(
        description="Detected user intent"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score (0-1)"
    )
    reasoning: Optional[str] = Field(
        None,
        description="Why this intent was chosen"
    )


class RouterAgent(BaseAgent):
    """
    Routes user queries to appropriate agents

    Strategy:
    1. Rule-based detection (fast, high precision)
    2. LLM-based classification (fallback, high recall)
    3. Hybrid scoring (combine both)
    """

    # Visualization keywords
    VIZ_KEYWORDS = [
        "show", "plot", "chart", "graph", "visualize", "visualization",
        "trend", "compare", "comparison", "display", "draw",
        "图表", "显示", "展示", "可视化", "趋势", "对比"
    ]

    # Report keywords
    REPORT_KEYWORDS = [
        "report", "summary", "analysis", "overview", "breakdown",
        "报告", "总结", "分析", "概览"
    ]

    def __init__(self, config: AgentConfig, llm_client: Optional[AnswerGenerator] = None):
        super().__init__(config)
        self.llm_client = llm_client

    async def execute(self, question: str, **kwargs) -> AgentResult:
        """
        Classify user intent

        Args:
            question: User question

        Returns:
            AgentResult with intent classification
        """
        # Validate
        self.validate_input(question=question)

        # Try rule-based first
        rule_result = self._rule_based_classification(question)

        if rule_result.confidence >= 0.8:
            # High confidence, use rule result
            return self._create_result(
                success=True,
                data=rule_result.dict(),
                metadata={"method": "rule_based"}
            )

        # Fallback to LLM if available
        if self.llm_client:
            llm_result = await self._llm_based_classification(question)
            return self._create_result(
                success=True,
                data=llm_result.dict(),
                metadata={"method": "llm_based"}
            )

        # Default to rule result
        return self._create_result(
            success=True,
            data=rule_result.dict(),
            metadata={"method": "rule_based_fallback"}
        )

    def validate_input(self, question: str, **kwargs) -> bool:
        """Validate input"""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        return True

    def _rule_based_classification(self, question: str) -> IntentClassification:
        """
        Rule-based intent detection

        Logic:
        - Check for visualization keywords
        - Check for report keywords
        - Default to QA
        """
        question_lower = question.lower()

        # Check visualization
        viz_score = sum(
            1 for kw in self.VIZ_KEYWORDS
            if kw in question_lower
        ) / len(self.VIZ_KEYWORDS)

        # Check report
        report_score = sum(
            1 for kw in self.REPORT_KEYWORDS
            if kw in question_lower
        ) / len(self.REPORT_KEYWORDS)

        # Decision logic
        if viz_score > 0:
            return IntentClassification(
                intent="visualization",
                confidence=min(0.6 + viz_score * 0.4, 0.95),
                reasoning=f"Detected visualization keywords in query"
            )
        elif report_score > 0:
            return IntentClassification(
                intent="report",
                confidence=min(0.6 + report_score * 0.4, 0.95),
                reasoning=f"Detected report keywords in query"
            )
        else:
            return IntentClassification(
                intent="qa",
                confidence=0.7,
                reasoning="No specific intent keywords detected, defaulting to QA"
            )

    async def _llm_based_classification(self, question: str) -> IntentClassification:
        """
        LLM-based intent classification (future enhancement)

        Uses DeepSeek Function Calling to classify intent
        """
        # TODO: Implement LLM classification
        # For now, fallback to rule-based
        return self._rule_based_classification(question)
```

**测试用例** (写在 `tests/test_agents/test_router.py`):

```python
import pytest
from app.agents.router import RouterAgent, AgentConfig


@pytest.fixture
def router():
    config = AgentConfig(
        name="router",
        description="Intent classification agent"
    )
    return RouterAgent(config)


@pytest.mark.asyncio
async def test_visualization_intent(router):
    """Test visualization intent detection"""
    result = await router.execute(question="Show me Apple revenue trend")

    assert result.success
    assert result.data["intent"] == "visualization"
    assert result.data["confidence"] > 0.6


@pytest.mark.asyncio
async def test_qa_intent(router):
    """Test QA intent detection"""
    result = await router.execute(question="What was Apple's revenue in 2024?")

    assert result.success
    assert result.data["intent"] == "qa"


@pytest.mark.asyncio
async def test_chinese_visualization(router):
    """Test Chinese visualization keywords"""
    result = await router.execute(question="显示苹果公司收入趋势")

    assert result.success
    assert result.data["intent"] == "visualization"
```

---

### Task 2: 实现数据提取工具 (3-4h)

#### 2.1 定义数据模型 (30min)

**文件**: `app/schemas/models.py` (追加)

```python
# Add to existing models.py

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


class ChartSeries(BaseModel):
    """Single data series for chart"""
    name: str = Field(description="Series name (e.g., 'Revenue', 'Net Income')")
    values: List[float] = Field(description="Data values")
    unit: Optional[str] = Field(None, description="Unit (e.g., 'Million USD')")


class ChartDataSchema(BaseModel):
    """
    Structured chart data extracted from documents

    This schema is used for LLM Function Calling
    to ensure structured output
    """
    title: str = Field(description="Chart title")
    x_label: str = Field(description="X-axis label (e.g., 'Year', 'Quarter')")
    y_label: str = Field(description="Y-axis label (e.g., 'Revenue', 'Amount')")
    x_values: List[Any] = Field(description="X-axis values (e.g., [2023, 2024, 2025])")
    series: List[ChartSeries] = Field(description="Data series")
    chart_type_hint: Optional[Literal["line", "bar", "pie", "scatter"]] = Field(
        None,
        description="Suggested chart type based on data"
    )
    data_source: Optional[str] = Field(
        None,
        description="Data source citation (e.g., '10-K 2024, Item 8')"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Apple Revenue Trend 2023-2025",
                "x_label": "Year",
                "y_label": "Revenue",
                "x_values": [2023, 2024, 2025],
                "series": [
                    {
                        "name": "Revenue",
                        "values": [383.285, 391.035, 401.610],
                        "unit": "Billion USD"
                    }
                ],
                "chart_type_hint": "line",
                "data_source": "10-K Reports, Item 8"
            }
        }


class VisualizationRequest(BaseModel):
    """Request for visualization endpoint"""
    question: str = Field(description="User question for visualization")
    chart_type: Literal["auto", "line", "bar", "pie", "grouped_bar"] = Field(
        default="auto",
        description="Chart type (auto = detect automatically)"
    )
    engine: Literal["plotly", "echarts"] = Field(
        default="plotly",
        description="Visualization engine"
    )
    session_id: Optional[str] = Field(
        None,
        description="Conversation session ID"
    )
    max_results: int = Field(
        default=10,
        ge=1, le=50,
        description="Max documents to retrieve"
    )


class VisualizationResponse(BaseModel):
    """Response from visualization endpoint"""
    session_id: str
    chart_html: str = Field(description="Chart HTML for embedding")
    chart_json: Dict[str, Any] = Field(description="Chart JSON for programmatic use")
    chart_data: Dict[str, Any] = Field(description="Raw extracted data")
    analysis: str = Field(description="Text analysis of the data")
    citations: List[Citation] = Field(description="Source citations")
    chart_type: str = Field(description="Actual chart type used")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (retrieval debug, etc.)"
    )
```

#### 2.2 实现 DataExtractor (2.5h)

**文件**: `app/tools/data_extractor.py`

```python
# LLM-based data extraction tool
import json
from typing import List, Optional
import logging

from app.schemas.models import (
    RetrievedDocument,
    ChartDataSchema,
    ChartSeries
)
from app.core.config import settings
from openai import OpenAI

logger = logging.getLogger(__name__)


class DataExtractor:
    """
    Extract structured chart data from text documents using LLM

    Uses DeepSeek Function Calling to ensure structured output
    """

    def __init__(self, llm_client: Optional[OpenAI] = None):
        """
        Initialize data extractor

        Args:
            llm_client: OpenAI-compatible client (DeepSeek)
        """
        if llm_client is None:
            # Create default client
            if settings.llm_provider == "deepseek":
                self.client = OpenAI(
                    api_key=settings.deepseek_api_key,
                    base_url=settings.deepseek_base_url
                )
                self.model = "deepseek-chat"
            else:
                raise ValueError("DataExtractor currently only supports DeepSeek")
        else:
            self.client = llm_client
            self.model = "deepseek-chat"

    def extract(
        self,
        question: str,
        documents: List[RetrievedDocument],
        max_context_length: int = 4000
    ) -> ChartDataSchema:
        """
        Extract structured chart data from documents

        Args:
            question: User question
            documents: Retrieved documents
            max_context_length: Max characters for context

        Returns:
            ChartDataSchema with extracted data

        Raises:
            ValueError: If extraction fails
        """
        # Build context from documents
        context = self._build_context(documents, max_context_length)

        # Prepare messages
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            },
            {
                "role": "user",
                "content": self._get_user_prompt(question, context)
            }
        ]

        # Prepare function definition
        function_def = {
            "name": "extract_chart_data",
            "description": "Extract numerical data from financial documents for chart visualization",
            "parameters": ChartDataSchema.model_json_schema()
        }

        try:
            # Call LLM with function calling
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[{
                    "type": "function",
                    "function": function_def
                }],
                tool_choice={"type": "function", "function": {"name": "extract_chart_data"}},
                temperature=0.1  # Low temperature for factual extraction
            )

            # Parse function call result
            tool_call = response.choices[0].message.tool_calls[0]
            chart_data_dict = json.loads(tool_call.function.arguments)

            # Validate and return
            chart_data = ChartDataSchema(**chart_data_dict)

            logger.info(f"Successfully extracted chart data: {chart_data.title}")
            return chart_data

        except Exception as e:
            logger.error(f"Data extraction failed: {str(e)}")
            raise ValueError(f"Failed to extract chart data: {str(e)}")

    def _build_context(
        self,
        documents: List[RetrievedDocument],
        max_length: int
    ) -> str:
        """
        Build context string from documents

        Prioritizes:
        1. Financial statements (tables)
        2. Higher scored documents
        """
        # Sort documents by priority
        sorted_docs = sorted(
            documents,
            key=lambda d: (
                # Prioritize financial statements
                d.metadata.get("item_type") == "financial_statements",
                # Then by score
                d.score
            ),
            reverse=True
        )

        context_parts = []
        current_length = 0

        for doc in sorted_docs:
            # Format document snippet
            snippet = self._format_document(doc)
            snippet_length = len(snippet)

            # Check if adding this would exceed limit
            if current_length + snippet_length > max_length:
                # Add partial if space allows
                remaining = max_length - current_length
                if remaining > 200:  # Only add if meaningful amount left
                    context_parts.append(snippet[:remaining] + "...\n")
                break

            context_parts.append(snippet)
            current_length += snippet_length

        return "\n".join(context_parts)

    def _format_document(self, doc: RetrievedDocument) -> str:
        """Format single document for context"""
        metadata = doc.metadata
        header = f"[{metadata.get('year')} - {metadata.get('section_title', 'Unknown')}]"

        # Truncate text if too long
        text = doc.text
        if len(text) > 1500:
            text = text[:1500] + "..."

        return f"{header}\n{text}\n{'='*60}\n"

    def _get_system_prompt(self) -> str:
        """Get system prompt for extraction"""
        return """You are a financial data extraction expert. Your task is to extract precise numerical data from financial reports to create charts.

**Critical Rules:**
1. Extract EXACT numbers from the documents - DO NOT fabricate or estimate
2. Include proper units (Million, Billion, etc.)
3. Maintain precision (decimals if present)
4. Cite the source section
5. If data is not found, indicate clearly
6. For multi-year questions, extract all available years
7. Be consistent with units across all values

**Example:**
If the document says "Revenue: $383.285 billion (2023), $391.035 billion (2024)",
Extract:
- x_values: [2023, 2024]
- series: [{"name": "Revenue", "values": [383.285, 391.035], "unit": "Billion USD"}]"""

    def _get_user_prompt(self, question: str, context: str) -> str:
        """Get user prompt for extraction"""
        return f"""Question: {question}

Financial Documents:
{context}

Extract the numerical data needed to answer this question and create a visualization.
Focus on being accurate and citing your sources."""
```

**测试用例** (`tests/test_tools/test_data_extractor.py`):

```python
import pytest
from unittest.mock import Mock, MagicMock
from app.tools.data_extractor import DataExtractor
from app.schemas.models import RetrievedDocument, ChartDataSchema


@pytest.fixture
def mock_llm_client():
    """Mock LLM client"""
    client = Mock()

    # Mock response
    mock_response = MagicMock()
    mock_tool_call = MagicMock()
    mock_tool_call.function.arguments = json.dumps({
        "title": "Apple Revenue 2023-2024",
        "x_label": "Year",
        "y_label": "Revenue",
        "x_values": [2023, 2024],
        "series": [{
            "name": "Revenue",
            "values": [383.285, 391.035],
            "unit": "Billion USD"
        }],
        "data_source": "10-K 2024"
    })
    mock_response.choices = [MagicMock(message=MagicMock(tool_calls=[mock_tool_call]))]
    client.chat.completions.create.return_value = mock_response

    return client


@pytest.fixture
def sample_documents():
    """Sample retrieved documents"""
    return [
        RetrievedDocument(
            doc_id="2024_item_8",
            text="Revenue: $391.035 billion for fiscal 2024",
            score=0.95,
            metadata={"year": 2024, "section_title": "Financial Statements"},
            retrieval_method="hybrid"
        ),
        RetrievedDocument(
            doc_id="2023_item_8",
            text="Revenue: $383.285 billion for fiscal 2023",
            score=0.90,
            metadata={"year": 2023, "section_title": "Financial Statements"},
            retrieval_method="hybrid"
        )
    ]


def test_extract_success(mock_llm_client, sample_documents):
    """Test successful data extraction"""
    extractor = DataExtractor(llm_client=mock_llm_client)

    result = extractor.extract(
        question="Show Apple revenue 2023-2024",
        documents=sample_documents
    )

    assert isinstance(result, ChartDataSchema)
    assert result.title == "Apple Revenue 2023-2024"
    assert len(result.series) == 1
    assert len(result.x_values) == 2
    assert result.series[0].values == [383.285, 391.035]


def test_build_context(mock_llm_client, sample_documents):
    """Test context building"""
    extractor = DataExtractor(llm_client=mock_llm_client)

    context = extractor._build_context(sample_documents, max_length=1000)

    assert "2024" in context
    assert "2023" in context
    assert "Revenue" in context
```

---

### Task 3: 实现 Plotly 图表生成器 (2-3h)

#### 3.1 实现 ChartGenerator (2h)

**文件**: `app/tools/chart_generator.py`

```python
# Plotly chart generation tool
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Literal
import logging

from app.schemas.models import ChartDataSchema

logger = logging.getLogger(__name__)


class ChartGenerator:
    """
    Generate interactive charts using Plotly

    Supported chart types:
    - line: Trend analysis
    - bar: Comparison
    - grouped_bar: Multi-series comparison
    - pie: Composition/breakdown
    """

    # Professional color palette
    COLORS = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
    ]

    def generate(
        self,
        data: ChartDataSchema,
        chart_type: Literal["line", "bar", "grouped_bar", "pie", "auto"] = "auto",
    ) -> go.Figure:
        """
        Generate chart from structured data

        Args:
            data: Chart data schema
            chart_type: Type of chart to generate

        Returns:
            Plotly Figure object
        """
        # Auto-detect chart type if needed
        if chart_type == "auto":
            chart_type = self._detect_chart_type(data)

        logger.info(f"Generating {chart_type} chart: {data.title}")

        # Route to specific generator
        if chart_type == "line":
            fig = self._generate_line_chart(data)
        elif chart_type == "bar":
            fig = self._generate_bar_chart(data)
        elif chart_type == "grouped_bar":
            fig = self._generate_grouped_bar_chart(data)
        elif chart_type == "pie":
            fig = self._generate_pie_chart(data)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        # Apply common layout
        self._apply_common_layout(fig, data)

        return fig

    def _detect_chart_type(self, data: ChartDataSchema) -> str:
        """
        Auto-detect appropriate chart type

        Logic:
        - Single series + multiple x-values → line (trend)
        - Multiple series + single x-value → bar (comparison)
        - Multiple series + multiple x-values → line (multi-trend)
        - Hint from LLM → use hint
        """
        # Check LLM hint first
        if data.chart_type_hint:
            return data.chart_type_hint

        num_series = len(data.series)
        num_x_values = len(data.x_values)

        # Decision tree
        if num_series == 1:
            if num_x_values >= 3:
                return "line"  # Trend over time
            else:
                return "bar"  # Simple comparison
        else:
            # Multiple series
            if num_x_values >= 3:
                return "line"  # Multi-line trend
            else:
                return "grouped_bar"  # Multi-series comparison

    def _generate_line_chart(self, data: ChartDataSchema) -> go.Figure:
        """Generate line chart for trends"""
        fig = go.Figure()

        for idx, series in enumerate(data.series):
            color = self.COLORS[idx % len(self.COLORS)]

            fig.add_trace(go.Scatter(
                x=data.x_values,
                y=series.values,
                mode='lines+markers',
                name=series.name,
                line=dict(width=3, color=color),
                marker=dict(size=10, color=color),
                hovertemplate=(
                    f"<b>{series.name}</b><br>"
                    f"{data.x_label}: %{{x}}<br>"
                    f"{data.y_label}: %{{y:.2f}} {series.unit or ''}<br>"
                    "<extra></extra>"
                )
            ))

        return fig

    def _generate_bar_chart(self, data: ChartDataSchema) -> go.Figure:
        """Generate bar chart for comparisons"""
        fig = go.Figure()

        for idx, series in enumerate(data.series):
            color = self.COLORS[idx % len(self.COLORS)]

            fig.add_trace(go.Bar(
                x=data.x_values,
                y=series.values,
                name=series.name,
                marker_color=color,
                hovertemplate=(
                    f"<b>{series.name}</b><br>"
                    f"{data.x_label}: %{{x}}<br>"
                    f"{data.y_label}: %{{y:.2f}} {series.unit or ''}<br>"
                    "<extra></extra>"
                )
            ))

        return fig

    def _generate_grouped_bar_chart(self, data: ChartDataSchema) -> go.Figure:
        """Generate grouped bar chart for multi-series comparison"""
        fig = go.Figure()

        for idx, series in enumerate(data.series):
            color = self.COLORS[idx % len(self.COLORS)]

            fig.add_trace(go.Bar(
                x=data.x_values,
                y=series.values,
                name=series.name,
                marker_color=color,
                hovertemplate=(
                    f"<b>{series.name}</b><br>"
                    f"{data.x_label}: %{{x}}<br>"
                    f"{data.y_label}: %{{y:.2f}} {series.unit or ''}<br>"
                    "<extra></extra>"
                )
            ))

        fig.update_layout(barmode='group')
        return fig

    def _generate_pie_chart(self, data: ChartDataSchema) -> go.Figure:
        """Generate pie chart for composition"""
        fig = go.Figure()

        # Use first series for pie chart
        series = data.series[0]

        fig.add_trace(go.Pie(
            labels=data.x_values,
            values=series.values,
            hole=0.3,  # Donut chart
            marker=dict(colors=self.COLORS),
            hovertemplate=(
                "<b>%{label}</b><br>"
                f"{series.name}: %{{value:.2f}} {series.unit or ''}<br>"
                "Percentage: %{percent}<br>"
                "<extra></extra>"
            )
        ))

        return fig

    def _apply_common_layout(self, fig: go.Figure, data: ChartDataSchema) -> None:
        """Apply common layout settings to all charts"""
        # Get y-axis label with unit
        y_label = data.y_label
        if data.series and data.series[0].unit:
            y_label = f"{y_label} ({data.series[0].unit})"

        fig.update_layout(
            title=dict(
                text=data.title,
                font=dict(size=22, family="Arial, sans-serif", color="#2c3e50"),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title=dict(
                text=data.x_label,
                font=dict(size=14, color="#34495e")
            ),
            yaxis_title=dict(
                text=y_label,
                font=dict(size=14, color="#34495e")
            ),
            template="plotly_white",
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12)
            ),
            margin=dict(l=60, r=40, t=80, b=60),
            font=dict(family="Arial, sans-serif", color="#2c3e50")
        )

        # Add gridlines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')

        # Add source annotation if available
        if data.data_source:
            fig.add_annotation(
                text=f"Source: {data.data_source}",
                xref="paper", yref="paper",
                x=0, y=-0.15,
                showarrow=False,
                font=dict(size=10, color="#7f8c8d"),
                xanchor='left'
            )
```

**测试用例** (`tests/test_tools/test_chart_generator.py`):

```python
import pytest
from app.tools.chart_generator import ChartGenerator
from app.schemas.models import ChartDataSchema, ChartSeries
import plotly.graph_objects as go


@pytest.fixture
def sample_chart_data():
    """Sample chart data"""
    return ChartDataSchema(
        title="Apple Revenue Trend 2023-2025",
        x_label="Year",
        y_label="Revenue",
        x_values=[2023, 2024, 2025],
        series=[
            ChartSeries(
                name="Revenue",
                values=[383.285, 391.035, 401.610],
                unit="Billion USD"
            )
        ]
    )


def test_generate_line_chart(sample_chart_data):
    """Test line chart generation"""
    generator = ChartGenerator()

    fig = generator.generate(sample_chart_data, chart_type="line")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].mode == "lines+markers"


def test_generate_bar_chart(sample_chart_data):
    """Test bar chart generation"""
    generator = ChartGenerator()

    fig = generator.generate(sample_chart_data, chart_type="bar")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"


def test_auto_detect_chart_type(sample_chart_data):
    """Test automatic chart type detection"""
    generator = ChartGenerator()

    # Single series + multiple x-values → should be line
    detected_type = generator._detect_chart_type(sample_chart_data)
    assert detected_type == "line"


def test_chart_has_title(sample_chart_data):
    """Test that generated chart has proper title"""
    generator = ChartGenerator()
    fig = generator.generate(sample_chart_data)

    assert fig.layout.title.text == "Apple Revenue Trend 2023-2025"
```

---

### Task 4: 实现 VisualizationAgent (2-3h)

**文件**: `app/agents/visualization_agent.py`

```python
# Visualization agent implementation
from typing import Optional
import logging

from app.agents.base import BaseAgent, AgentConfig, AgentResult
from app.retrieve.hybrid_retriever import HybridRetriever
from app.tools.data_extractor import DataExtractor
from app.tools.chart_generator import ChartGenerator
from app.schemas.models import ChartDataSchema, VisualizationResponse, Citation
from app.generate.answer_generator import AnswerGenerator

logger = logging.getLogger(__name__)


class VisualizationAgent(BaseAgent):
    """
    Agent for financial data visualization

    Pipeline:
    1. Retrieve relevant documents
    2. Extract structured data with LLM
    3. Generate chart
    4. Generate analysis text
    5. Extract citations
    """

    def __init__(
        self,
        config: AgentConfig,
        retriever: HybridRetriever,
        data_extractor: DataExtractor,
        chart_generator: ChartGenerator,
        answer_generator: AnswerGenerator
    ):
        super().__init__(config)
        self.retriever = retriever
        self.data_extractor = data_extractor
        self.chart_generator = chart_generator
        self.answer_generator = answer_generator

    async def execute(
        self,
        question: str,
        chart_type: str = "auto",
        session_id: Optional[str] = None,
        max_results: int = 10,
        **kwargs
    ) -> AgentResult:
        """
        Execute visualization pipeline

        Args:
            question: User question
            chart_type: Chart type or "auto"
            session_id: Session ID for conversation
            max_results: Max documents to retrieve

        Returns:
            AgentResult with VisualizationResponse
        """
        try:
            # Validate input
            self.validate_input(question=question)

            # Step 1: Retrieve documents
            logger.info(f"Retrieving documents for: {question}")
            retrieved_docs, debug_info = self.retriever.retrieve(
                query=question,
                k=max_results
            )

            if not retrieved_docs:
                return self._create_result(
                    success=False,
                    data={},
                    error="No relevant documents found"
                )

            # Step 2: Extract structured data
            logger.info("Extracting chart data from documents")
            chart_data = self.data_extractor.extract(
                question=question,
                documents=retrieved_docs
            )

            # Step 3: Generate chart
            logger.info(f"Generating {chart_type} chart")
            fig = self.chart_generator.generate(
                data=chart_data,
                chart_type=chart_type
            )

            # Step 4: Generate analysis text
            analysis = self._generate_analysis(chart_data, question)

            # Step 5: Extract citations
            citations = self._extract_citations(retrieved_docs, chart_data)

            # Prepare response
            response = VisualizationResponse(
                session_id=session_id or "default",
                chart_html=fig.to_html(include_plotlyjs='cdn'),
                chart_json=fig.to_dict(),
                chart_data=chart_data.dict(),
                analysis=analysis,
                citations=citations,
                chart_type=self.chart_generator._detect_chart_type(chart_data) if chart_type == "auto" else chart_type,
                metadata={
                    "retrieval_debug": debug_info,
                    "num_documents": len(retrieved_docs)
                }
            )

            return self._create_result(
                success=True,
                data=response.dict()
            )

        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}", exc_info=True)
            return self._create_result(
                success=False,
                data={},
                error=str(e)
            )

    def validate_input(self, question: str, **kwargs) -> bool:
        """Validate input"""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        return True

    def _generate_analysis(self, chart_data: ChartDataSchema, question: str) -> str:
        """
        Generate text analysis of chart data

        Simple rule-based for MVP, can be enhanced with LLM
        """
        try:
            # Get primary series
            if not chart_data.series:
                return "No data available for analysis."

            series = chart_data.series[0]
            values = series.values

            if len(values) < 2:
                return f"{series.name}: {values[0]:.2f} {series.unit or ''}"

            # Calculate basic stats
            start_value = values[0]
            end_value = values[-1]
            change = end_value - start_value
            pct_change = (change / start_value * 100) if start_value != 0 else 0

            # Determine trend
            if change > 0:
                trend = "increased"
            elif change < 0:
                trend = "decreased"
            else:
                trend = "remained stable"

            # Build analysis
            analysis = f"{series.name} {trend} from {start_value:.2f} {series.unit or ''} "
            analysis += f"({chart_data.x_values[0]}) to {end_value:.2f} {series.unit or ''} "
            analysis += f"({chart_data.x_values[-1]}), "
            analysis += f"representing a {abs(pct_change):.1f}% {'increase' if change > 0 else 'decrease'}."

            # Multi-series analysis
            if len(chart_data.series) > 1:
                analysis += f"\n\nThis chart compares {len(chart_data.series)} metrics: "
                analysis += ", ".join([s.name for s in chart_data.series]) + "."

            return analysis

        except Exception as e:
            logger.error(f"Analysis generation failed: {e}")
            return "Analysis unavailable."

    def _extract_citations(
        self,
        documents: list,
        chart_data: ChartDataSchema
    ) -> list[Citation]:
        """Extract citations from retrieved documents"""
        citations = []
        seen = set()

        for doc in documents[:5]:  # Top 5 documents
            metadata = doc.metadata

            # Create citation key to avoid duplicates
            citation_key = f"{metadata.get('year')}_{metadata.get('section_title')}"

            if citation_key not in seen:
                citations.append(Citation(
                    year=metadata.get('year'),
                    section_title=metadata.get('section_title', 'Unknown'),
                    chunk_id=metadata.get('chunk_id')
                ))
                seen.add(citation_key)

        return citations
```

---

### Task 5: 新增 API 端点 (1-2h)

**文件**: `app/api/main.py` (修改)

```python
# Add to existing main.py

from app.agents.orchestrator import AgentOrchestrator
from app.agents.router import RouterAgent, AgentConfig
from app.agents.visualization_agent import VisualizationAgent
from app.tools.data_extractor import DataExtractor
from app.tools.chart_generator import ChartGenerator
from app.schemas.models import VisualizationRequest, VisualizationResponse

# ... existing imports ...


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management with agent initialization"""
    logger.info("Starting Fin-Agent API with Multi-Agent support...")

    # Existing initializations
    bm25_retriever = BM25Retriever(...)
    chroma_retriever = ChromaRetriever(...)
    hybrid_retriever = HybridRetriever(...)
    answer_generator = AnswerGenerator(...)
    conversation_manager = ConversationManager(...)

    # 🆕 Initialize agents
    router_agent = RouterAgent(
        config=AgentConfig(
            name="router",
            description="Intent classification agent"
        )
    )

    data_extractor = DataExtractor()
    chart_generator = ChartGenerator()

    visualization_agent = VisualizationAgent(
        config=AgentConfig(
            name="visualizer",
            description="Financial data visualization agent"
        ),
        retriever=hybrid_retriever,
        data_extractor=data_extractor,
        chart_generator=chart_generator,
        answer_generator=answer_generator
    )

    # 🆕 Initialize orchestrator
    orchestrator = AgentOrchestrator(
        router=router_agent,
        visualization_agent=visualization_agent,
        # qa_agent will be added in next sprint
    )

    # Store in app state
    app.state.bm25_retriever = bm25_retriever
    app.state.chroma_retriever = chroma_retriever
    app.state.hybrid_retriever = hybrid_retriever
    app.state.answer_generator = answer_generator
    app.state.conversation_manager = conversation_manager
    app.state.orchestrator = orchestrator  # 🆕

    logger.info("✅ All agents initialized successfully")

    yield

    logger.info("Shutting down Fin-Agent API...")


# 🆕 New endpoint: /visualize
@app.post("/visualize", response_model=VisualizationResponse)
async def visualize_data(request: VisualizationRequest):
    """
    Generate financial data visualization

    This endpoint:
    1. Retrieves relevant financial documents
    2. Extracts structured numerical data using LLM
    3. Generates interactive chart (Plotly)
    4. Provides text analysis
    5. Returns citations

    Example:
        POST /visualize
        {
            "question": "Show Apple revenue trend 2023-2025",
            "chart_type": "auto",
            "session_id": "uuid"
        }
    """
    try:
        # Get orchestrator
        orchestrator: AgentOrchestrator = app.state.orchestrator
        conversation_manager = app.state.conversation_manager

        # Ensure session
        session_id = conversation_manager.ensure_session(request.session_id)

        # Execute visualization agent
        result = await orchestrator.execute_visualization(
            question=request.question,
            chart_type=request.chart_type,
            session_id=session_id,
            max_results=request.max_results
        )

        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=result.error or "Visualization failed"
            )

        # Convert to response model
        response = VisualizationResponse(**result.data)

        # Save to conversation history
        conversation_manager.append_turn(
            session_id=session_id,
            question=request.question,
            answer=response.analysis,
            citations=response.citations
        )

        return response

    except Exception as e:
        logger.error(f"Visualization endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Update root endpoint to show new endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Fin-Agent API",
        "version": "2.0.0",  # Updated version
        "description": "Multi-Agent Financial Report RAG System with Visualization",
        "endpoints": {
            "health": "GET /health",
            "index_status": "GET /index/status",
            "build_index": "POST /index/build",
            "query": "POST /query",
            "visualize": "POST /visualize",  # 🆕
            "report": "POST /report",
            "docs": "GET /docs"
        }
    }
```

**简化版 Orchestrator** (`app/agents/orchestrator.py`):

```python
# Agent orchestration logic
from typing import Optional
import logging

from app.agents.base import AgentResult
from app.agents.router import RouterAgent
from app.agents.visualization_agent import VisualizationAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Orchestrates multiple agents

    For Sprint 1, supports:
    - Visualization agent
    - (QA agent to be added in Sprint 2)
    """

    def __init__(
        self,
        router: RouterAgent,
        visualization_agent: VisualizationAgent,
        qa_agent: Optional[Any] = None  # To be added
    ):
        self.router = router
        self.visualization_agent = visualization_agent
        self.qa_agent = qa_agent

    async def execute_visualization(
        self,
        question: str,
        chart_type: str = "auto",
        session_id: Optional[str] = None,
        max_results: int = 10
    ) -> AgentResult:
        """
        Execute visualization agent directly

        For Sprint 1, this is a direct call.
        In Sprint 2, this will be routed through router agent.
        """
        logger.info(f"Executing visualization for: {question}")

        result = await self.visualization_agent.execute(
            question=question,
            chart_type=chart_type,
            session_id=session_id,
            max_results=max_results
        )

        return result
```

---

### Task 6: 配置和依赖更新 (30min)

#### 6.1 更新 requirements.txt

```bash
# Add to requirements.txt
plotly>=5.18.0
kaleido>=0.2.1  # For static image export (optional)
```

#### 6.2 更新配置

**文件**: `app/core/config.py` (追加)

```python
# Add to Settings class

class Settings(BaseSettings):
    # ... existing settings ...

    # 🆕 Visualization settings
    visualization_enabled: bool = True
    default_chart_engine: str = "plotly"
    max_chart_data_points: int = 1000
    chart_default_width: int = 800
    chart_default_height: int = 600
```

---

### Task 7: 集成测试 (1-2h)

**文件**: `tests/test_integration/test_visualization_flow.py`

```python
# End-to-end visualization test
import pytest
from fastapi.testclient import TestClient
from app.api.main import app


client = TestClient(app)


def test_visualize_endpoint_success():
    """Test successful visualization request"""
    response = client.post("/visualize", json={
        "question": "Show Apple revenue trend from 2023 to 2025",
        "chart_type": "auto",
        "engine": "plotly"
    })

    assert response.status_code == 200
    data = response.json()

    assert "chart_html" in data
    assert "chart_json" in data
    assert "analysis" in data
    assert "citations" in data
    assert len(data["citations"]) > 0


def test_visualize_invalid_request():
    """Test invalid request handling"""
    response = client.post("/visualize", json={
        "question": "",  # Empty question
    })

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_full_visualization_pipeline():
    """Test complete pipeline from question to chart"""
    # This would require mocking or using test data
    pass
```

---

## 📊 验收标准

### 功能验收

| 功能 | 验收标准 | 状态 |
|------|---------|------|
| **Router Agent** | 能识别可视化意图，准确率 >80% | ⬜ |
| **Data Extraction** | 能从文本提取结构化数据，准确率 >90% | ⬜ |
| **Chart Generation** | 能生成 line 图表，显示正常 | ⬜ |
| **API Endpoint** | `/visualize` 端点返回正确响应 | ⬜ |
| **Error Handling** | 无数据或错误时有友好提示 | ⬜ |
| **Citations** | 包含数据来源引用 | ⬜ |

### 代码质量验收

- ⬜ 所有新增代码有类型注解
- ⬜ 关键函数有 docstring
- ⬜ 单元测试覆盖率 >80%
- ⬜ 集成测试通过
- ⬜ 代码符合 PEP 8 规范

### 性能验收

- ⬜ 端到端响应时间 <10s
- ⬜ 数据提取准确率 >90%
- ⬜ 无内存泄漏

---

## 🎯 示例测试场景

### 场景 1: 简单趋势图

**输入**:
```json
{
  "question": "Show me Apple's revenue from 2023 to 2025",
  "chart_type": "auto"
}
```

**预期输出**:
- Chart type: line
- X-axis: [2023, 2024, 2025]
- Y-axis: Revenue values
- Analysis: "Revenue increased from X to Y..."
- Citations: 10-K reports

### 场景 2: 中文问题

**输入**:
```json
{
  "question": "显示苹果公司2023到2025年的收入趋势",
  "chart_type": "auto"
}
```

**预期输出**:
- 自动翻译为英文
- 生成英文图表
- 分析文本为英文

### 场景 3: 无数据场景

**输入**:
```json
{
  "question": "Show Apple revenue in 2030",
  "chart_type": "auto"
}
```

**预期输出**:
- HTTP 200 (不是 500)
- Error message: "No data found for 2030"

---

## ⚠️ 风险和应对

### 风险 1: LLM 数据提取准确率低

**应对**:
- 优化 prompt 工程
- 增加示例 (few-shot learning)
- 添加数据验证逻辑

### 风险 2: DeepSeek Function Calling 不稳定

**应对**:
- 实现 retry 机制
- 提供降级方案（正则表达式提取）

### 风险 3: 图表渲染性能问题

**应对**:
- 限制数据点数量
- 实现数据采样

### 风险 4: 集成现有代码出现冲突

**应对**:
- 充分的单元测试
- 逐步集成，小步快跑

---

## 📅 实施时间表

| 任务 | 预计时间 | 开始日期 | 完成日期 |
|------|---------|---------|---------|
| Task 1: Agent 基础架构 | 4-5h | Day 1 | Day 1 |
| Task 2: DataExtractor | 3-4h | Day 1-2 | Day 2 |
| Task 3: ChartGenerator | 2-3h | Day 2 | Day 2 |
| Task 4: VisualizationAgent | 2-3h | Day 2-3 | Day 3 |
| Task 5: API 端点 | 1-2h | Day 3 | Day 3 |
| Task 6: 配置更新 | 30min | Day 3 | Day 3 |
| Task 7: 集成测试 | 1-2h | Day 3-4 | Day 4 |
| **总计** | **14-20h** | **Day 1** | **Day 4** |

---

## 🚀 下一步行动

### 立即开始

1. **创建分支**
   ```bash
   git checkout -b feature/sprint1-multi-agent-viz
   ```

2. **创建模块结构**
   ```bash
   mkdir -p app/agents app/tools tests/test_agents tests/test_tools
   ```

3. **安装依赖**
   ```bash
   pip install plotly kaleido
   ```

4. **开始实现 Task 1**

---

## 📝 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|---------|
| 2026-04-06 | 1.0 | 初始计划文档创建 |

---

**文档作者**: Pike
**最后更新**: 2026-04-06
**状态**: ✅ Ready for Implementation
