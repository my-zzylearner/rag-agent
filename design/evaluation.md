# RAG 知识库评估体系

## 概述

离线评估脚本 `eval/evaluate.py`，自动从知识库文档生成问答对，通过 RAGAS 框架评估 RAG pipeline 质量，输出报告到 `eval/report.md`。

## 评估指标

| 指标 | 含义 | 计算方式 | 目标值 |
|------|------|----------|--------|
| **Faithfulness** | 回答忠于检索内容的程度 | 回答中每个陈述是否能从检索上下文中推导出来 | ≥ 0.9 |
| **Answer Relevancy** | 回答与问题的相关程度 | 从回答逆向生成问题，与原问题的语义相似度 | ≥ 0.7 |
| **Context Recall** | 检索内容覆盖标准答案的程度 | 标准答案中每个陈述是否能从检索上下文中找到支撑 | ≥ 0.7 |

### 指标解读

- **Faithfulness 高但 Context Recall 低**：模型回答诚实（没有幻觉），但检索没找到正确内容，说明**检索召回率是瓶颈**
- **Answer Relevancy 低**：通常是因为检索失败后模型回答"未找到相关内容"，这种回答和问题语义无关，拉低均值
- 三个指标同时低：检索和生成都有问题

## 运行原理

```
data/docs/*.md
    ↓ LLM 自动生成问答对（每文档取前3个chunk）
eval/questions.json（可手动修改）
    ↓ RAG pipeline（retrieve → LLM 生成回答）
    ↓ RAGAS evaluate（LLM 评估 + SentenceTransformer embedding）
eval/report.md
```

### 关键设计决策

- **问题生成**：每个 chunk 用 LLM 生成一对问答，保存到 `questions.json`，支持手动修改后用 `--skip-gen` 跳过重新生成
- **RAG pipeline**：复用项目的 `retrieve()` + LLM，走 `_build_candidates()` 的 fallback 逻辑
- **RAGAS LLM**：读 `LLM_JUDGE` 环境变量（默认 `qwen-turbo`），独立于主模型，避免主模型额度影响评估
- **Embedding**：使用本地 `SentenceTransformer`（all-MiniLM-L6-v2），避免百炼 embedding API 格式不兼容

### 运行命令

```bash
# 首次运行（生成问题 + 评估）
python eval/evaluate.py

# 跳过问题生成（手动修改 questions.json 后）
python eval/evaluate.py --skip-gen

# 调整检索条数
python eval/evaluate.py --skip-gen --top-k 6
```

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_JUDGE` | RAGAS 评估使用的模型 | `bailian/qwen-turbo` |
| `LLM` | RAG pipeline 使用的模型 | 同主应用 |
| `LLM_FALLBACK` | RAG pipeline fallback 模型 | 同主应用 |

## 历史评估结果

### 2026-04-12（基线）

| 指标 | 值 | 状态 |
|------|-----|------|
| Faithfulness | 0.882 | ✅ 良好 |
| Answer Relevancy | 0.120 | ❌ 偏低 |
| Context Recall | 0.263 | ❌ 偏低 |

**样本数**：19 题（9 篇文档，每篇取前 3 个 chunk）

**问题分析**：
- Faithfulness 较高，说明模型回答诚实，没有明显幻觉
- Context Recall 和 Answer Relevancy 偏低的根本原因：**向量检索召回率不足**，大量问题（Q7-Q19）检索不到对应文档内容
- 典型失败模式：协同过滤、向量数据库等主题的问题，检索返回了无关内容，模型如实回答"未找到相关内容"

**改进方向**：
- 混合检索（BM25 + 向量 + RRF）可显著提升 Context Recall
- 增加知识库文档覆盖度

## 待解决问题

- Answer Relevancy 计算依赖 embedding 语义相似度，当前用本地 SentenceTransformer，精度有限
- 评估样本仅覆盖文档前几个 chunk，长文档后半部分未被评估
