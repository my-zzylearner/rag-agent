# 🔍 AI Search Agent

一个结合 **RAG（检索增强生成）** 和 **Web Search** 的智能问答 Agent，由阿里百炼 Qwen 驱动，展示 AI 搜索工程的核心技术链路。

**[👉 在线体验 Demo](https://rag-search-agent.streamlit.app)**

---

## 架构设计

```
用户提问
    ↓
Agent（阿里百炼 Qwen · OpenAI 兼容接口 + Tool Calling）
    ├── 📚 Tool 1: 本地知识库检索（RAG）
    │       文档 → 语义 Chunking → Embedding → ChromaDB → 向量检索
    └── 🌐 Tool 2: 实时网络搜索（Tavily）
            ↓ 自动内化
        搜索结果 → LLM 提炼 → 结构化 Markdown → ChromaDB + data/docs/
    ↓
融合多路结果 → 流式生成回答 + 标注来源
```

## 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| LLM | 阿里百炼 Qwen（可配置） | OpenAI 兼容接口，Tool Calling，支持多模型 fallback |
| Embedding | all-MiniLM-L6-v2 | 本地运行，无需 API |
| 向量库 | ChromaDB | cosine 相似度检索，语义 chunk 切分 |
| 网络搜索 | Tavily API | 实时互联网搜索 |
| 前端 | Streamlit | 流式输出，展示 Agent 思考过程和检索来源 |

## 核心功能

**检索与生成**
- **智能工具路由**：实时信息（天气/股价）直接走网络搜索，专业知识优先查本地知识库，无结果时自动降级
- **流式输出**：LLM 回答逐 token 流式渲染，▌光标指示生成中
- **语义 Chunking**：按段落边界切分（MAX_CHUNK_SIZE=800），避免截断表格和代码块
- **来源溯源**：参考来源展示相关度评分，过滤低相关度结果，query 关键词高亮

**知识自动内化**
- 网络搜索结果经 LLM 提炼为结构化 Markdown，异步写入 `data/docs/`
- frontmatter 动态路由，自动归类到最相关的知识文档
- ChromaDB 实时重索引，下次同类问题直接本地检索
- 侧边栏展示最近内化记录

**工程能力**
- **多模型 fallback**：`LLM_FALLBACK` 环境变量配置备用模型，额度不足/限流时自动切换并提示
- **停止控制**：Agent 运行中可随时中止，已有中间结果保留展示
- **运行时调参**：侧边栏滑块实时调整检索条数和工具调用轮次，无需重启
- **统一日志**：关键路径 ERROR 日志，`DEBUG=true` 输出详细链路日志
- **.env 热重载**：修改配置后下次提问即生效，无需重启

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
# 编辑 .env，填入 DASHSCOPE_API_KEY 和 TAVILY_API_KEY

# 4. 启动
streamlit run app.py
```

> **首次启动**：sentence-transformers 模型（~90MB）自动下载，约需 2~3 分钟，后续秒开。

## 项目结构

```
rag-agent/
├── app.py                          # Streamlit 前端
├── agent/
│   ├── agent.py                    # Agent 主循环（Tool Calling + 多模型 fallback + 流式输出）
│   ├── tools.py                    # 工具定义与执行（含知识内化触发）
│   └── logger.py                   # 结构化日志（trace_id、JSON 格式）
├── rag/
│   ├── indexer.py                  # 文档加载、语义 Chunking、Embedding、入库
│   ├── retriever.py                # 向量检索
│   └── knowledge_internalizer.py  # LLM 提炼 + 文档增量写入 + 质量过滤
├── eval/
│   ├── evaluate.py                 # RAGAS 离线评估脚本（自动生成问答对 + 评测报告）
│   ├── questions.json              # 评测问答对
│   └── report.md                   # 最新评测报告
├── utils/
│   ├── logger.py                   # 统一日志模块（RotatingFileHandler + stderr）
│   └── gist_store.py               # GitHub Gist 持久化（访问统计 + 留言板）
├── tests/                          # 单元测试
├── data/docs/                      # 知识库文档（含自动内化生成的文档）
├── logs/                           # 运行日志（app.log，按大小轮转）
└── design/                         # 方案设计文档
```

## 知识库内容

- RAG 技术原理与 Chunking 策略
- 向量数据库选型对比（Chroma/Pinecone/Milvus/Qdrant）
- 搜索排序算法（BM25、向量检索、混合检索、Reranking）
- LLM Agent 架构与 Harness 工程实践

## 测试

```bash
# 快速测试（跳过需要模型加载的用例）
venv/bin/pytest tests/ -m "not slow"

# 完整测试（含 embedding 检索链路）
venv/bin/pytest tests/ -m slow
```
