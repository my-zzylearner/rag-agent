# 🔍 AI Search Agent

一个结合 **RAG（检索增强生成）** 和 **Web Search** 的智能问答 Agent，展示 AI 搜索工程的核心技术链路。

**[👉 在线体验 Demo]()**  ← 部署后填入链接

---

## 架构设计

```
用户提问
    ↓
Agent（文心 ERNIE · OpenAI 兼容接口 + Tool Calling）
    ├── 📚 Tool 1: 本地知识库检索（RAG）
    │       文档 → Chunking → Embedding → ChromaDB → 向量检索
    └── 🌐 Tool 2: 实时网络搜索（Tavily）
            ↓ 自动内化
        搜索结果 → Chunking → Embedding → ChromaDB（web_cache）
    ↓
融合多路结果 → 生成回答 + 标注来源
```

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| LLM | 文心 ERNIE-Speed-Pro | OpenAI 兼容接口，Tool Calling |
| Embedding | all-MiniLM-L6-v2 | 本地运行，无需 API |
| 向量库 | ChromaDB | 持久化存储，cosine 相似度检索 |
| 网络搜索 | Tavily API | 实时互联网搜索 |
| 前端 | Streamlit | 展示 Agent 思考过程和检索来源 |

## 核心技术点

- **Chunking 策略**：chunk_size=512，overlap=64，保留跨块上下文
- **多路召回**：知识库语义检索 + 实时网络搜索，互补覆盖
- **智能工具路由**：实时信息直接走网络搜索，专业知识优先查本地知识库，知识库无结果时自动降级到网络
- **知识自动内化**：网络搜索结果自动写入 ChromaDB，同类问题下次直接本地检索，upsert 去重
- **停止控制**：Agent 思考中可随时中止，已有中间结果保留展示
- **来源溯源**：每条回答标注检索片段来源，提高可信度
- **循环保护**：最大工具调用轮次限制，防止死循环

## 本地运行

```bash
# 1. 克隆项目
git clone <repo_url>
cd rag-agent

# 2. 创建虚拟环境并安装依赖
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 QIANFAN_API_KEY 和 TAVILY_API_KEY

# 4. 启动
streamlit run app.py
```

> **注意事项**
> - **首次启动需等待 2~3 分钟**：sentence-transformers 模型（~90MB）会自动下载到本地缓存，后续启动秒开
> - **首次提问需等待约 30 秒**：知识库向量化建立索引，完成后页面会提示"知识库已建立"
> - 后续所有操作均为正常响应速度，无需再等待

## 项目结构

```
rag-agent/
├── app.py              # Streamlit 前端（含停止按钮）
├── agent/
│   ├── agent.py        # Agent 主循环（Tool Calling + 停止控制）
│   └── tools.py        # 工具定义与执行（含知识内化）
├── rag/
│   ├── indexer.py      # 文档加载、Chunking、Embedding、入库、增量写入
│   └── retriever.py    # 向量检索
├── data/docs/          # 知识库文档（AI 搜索相关）
└── design/             # 方案设计文档
```

## 知识库内容

- RAG 技术原理与优化策略
- 向量数据库选型对比（Chroma/Pinecone/Milvus/Qdrant）
- 搜索排序算法（BM25、向量检索、混合检索、Reranking）
- LLM Agent 架构与 Function Calling 机制

## 测试

```bash
# 快速测试（默认，跳过需要模型加载的用例，秒级完成）
venv/bin/pytest tests/ -m "not slow"

# 完整测试（包含 embedding 检索链路，首次运行需下载模型）
venv/bin/pytest tests/ -m slow
```

测试覆盖：
- Chunking 逻辑（overlap、边界情况）
- 知识库写入与去重
- 向量检索结果格式与分数范围
- Agent 工具描述格式（Function Calling 规范）
- 工具异常入参处理

## 面试延伸

> 如果这个项目要上生产，你会怎么改？

- **向量库**：Chroma → Qdrant/Milvus，支持亿级向量
- **检索**：纯向量 → 混合检索（BM25 + 向量 + RRF 融合）
- **Reranking**：加 Cross-Encoder 对召回结果重排
- **知识库更新**：增量索引，避免全量重建
- **评估**：接入 RAGAS 框架，持续监控 Faithfulness 和 Answer Relevancy
