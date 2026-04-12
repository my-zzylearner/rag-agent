"""
RAG 知识库离线评估脚本

流程：
1. 从 data/docs/ 读取文档，按 chunk 切分
2. 用 LLM 为每个 chunk 自动生成问题（可手动修改 eval/questions.json）
3. 用 RAG pipeline 检索并生成回答
4. 用 RAGAS 评估 Faithfulness / Answer Relevancy / Context Recall
5. 输出报告到 eval/report.md

用法：
  cd /path/to/rag-agent
  source venv/bin/activate
  python eval/evaluate.py [--skip-gen]  # --skip-gen 跳过问题生成，直接用已有 questions.json
"""
import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(Path(__file__).parent.parent / ".env")

# 禁用 chromadb 遥测
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI  # noqa: E402, F401
from rag.retriever import retrieve  # noqa: E402
from rag.indexer import load_documents, chunk_text  # noqa: E402

EVAL_DIR = Path(__file__).parent
QUESTIONS_FILE = EVAL_DIR / "questions.json"
REPORT_FILE = EVAL_DIR / "report.md"
DOCS_DIR = Path(__file__).parent.parent / "data" / "docs"

# 每个文档生成的问题数
QUESTIONS_PER_DOC = 3


def _get_candidates():
    """复用 agent 的候选列表。"""
    from agent.agent import _build_candidates
    return _build_candidates()


def _call_with_fallback(candidates: list, messages: list, temperature: float = 0) -> str:
    """遍历候选模型，遇到额度/限流错误自动切换，返回回答内容。"""
    from agent.agent import _should_fallback
    last_exc = None
    for client, model, label in candidates:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if _should_fallback(e):
                print(f"  ⚠️ {label} 失败，切换下一个模型...")
                last_exc = e
                continue
            raise
    raise RuntimeError(f"所有候选模型均失败，最后错误: {last_exc}")


def generate_questions(client, model: str) -> "list[dict]":
    """从文档自动生成问答对，返回 [{"question": ..., "ground_truth": ..., "source": ...}]"""
    print("📝 正在从文档生成问题...")
    docs = load_documents(str(DOCS_DIR))
    qa_pairs = []

    for doc in docs:
        chunks = chunk_text(doc["text"], doc["source"])
        # 每个文档取前几个 chunk 生成问题
        sample_chunks = chunks[:QUESTIONS_PER_DOC]
        for chunk in sample_chunks:
            if len(chunk["text"].strip()) < 100:
                continue
            prompt = f"""根据以下文本内容，生成一个具体的问题和对应的标准答案。
要求：
- 问题要有实际意义，能用文本内容直接回答
- 答案要简洁准确，直接来自文本
- 返回 JSON 格式：{{"question": "...", "answer": "..."}}

文本内容：
{chunk["text"][:800]}"""
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                raw = resp.choices[0].message.content.strip()
                # 提取 JSON
                if "```" in raw:
                    raw = raw.split("```")[1].lstrip("json").strip()
                data = json.loads(raw)
                qa_pairs.append({
                    "question": data["question"],
                    "ground_truth": data["answer"],
                    "source": doc["source"],
                })
                print(f"  ✓ {doc['source']}: {data['question'][:50]}...")
            except Exception as e:
                print(f"  ✗ {doc['source']} 生成失败: {e}")

    return qa_pairs


def run_rag(question: str, top_k: int = 4) -> "tuple[str, list[str]]":
    """用 RAG pipeline 回答问题，返回 (answer, contexts)"""
    candidates = _get_candidates()
    chunks = retrieve(question, top_k=top_k)
    contexts = [c["text"] for c in chunks]

    if not contexts:
        return "未找到相关内容", []

    context_text = "\n\n---\n\n".join(contexts)
    prompt = f"""根据以下参考内容回答问题，只使用参考内容中的信息，不要编造。

参考内容：
{context_text}

问题：{question}"""

    answer = _call_with_fallback(candidates, [{"role": "user", "content": prompt}])
    return answer, contexts


class _STEmbeddings:
    """用项目已有的 SentenceTransformer 包装成 RAGAS Embeddings 接口。"""

    def __init__(self):
        from rag.indexer import get_embedder
        self._model = get_embedder()

    def embed_query(self, text: str) -> "list[float]":
        return self._model.encode([text]).tolist()[0]

    def embed_documents(self, texts: "list[str]") -> "list[list[float]]":
        return self._model.encode(texts).tolist()


def evaluate(qa_pairs: "list[dict]") -> "tuple":
    """用 RAGAS 评估，返回 (result, rows)"""
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import FakeEmbeddings  # noqa: F401

    print("\n🔍 正在运行 RAG pipeline...")
    rows = []
    for i, qa in enumerate(qa_pairs):
        print(f"  [{i+1}/{len(qa_pairs)}] {qa['question'][:60]}...")
        answer, contexts = run_rag(qa["question"])
        rows.append({
            "question": qa["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": qa["ground_truth"],
        })

    dataset = Dataset.from_list(rows)

    # RAGAS 内部并发调用，无法走 agent fallback，优先用 LLM_JUDGE 指定的评估模型
    from agent.agent import PROVIDERS
    judge_llm = os.getenv("LLM_JUDGE") or os.getenv("LLM", "bailian/qwen-turbo")
    judge_provider = judge_llm.split("/")[0].lower()
    judge_model = judge_llm.split("/", 1)[1] if "/" in judge_llm else "qwen-turbo"
    judge_cfg = PROVIDERS.get(judge_provider, PROVIDERS["bailian"])
    judge_api_key = os.getenv(judge_cfg["env_var"], "")
    judge_base_url = judge_cfg["base_url"]

    ragas_llm = LangchainLLMWrapper(ChatOpenAI(
        model=judge_model,
        openai_api_key=judge_api_key,
        openai_api_base=judge_base_url,
        temperature=0,
    ))

    # 用本地 SentenceTransformer，避免百炼 embedding API 格式不兼容
    print("📦 加载本地 Embedding 模型...")
    st_embeddings = _STEmbeddings()

    class _LCWrapper:
        """最小化 LangChain Embeddings 接口，供 LangchainEmbeddingsWrapper 使用。"""
        def embed_query(self, text):
            return st_embeddings.embed_query(text)

        def embed_documents(self, texts):
            return st_embeddings.embed_documents(texts)

    ragas_embeddings = LangchainEmbeddingsWrapper(_LCWrapper())

    print("\n📊 正在运行 RAGAS 评估...")
    result = ragas_evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )
    return result, rows


def write_report(result, rows: "list[dict]", qa_pairs: "list[dict]") -> None:
    """写入 Markdown 报告"""
    from datetime import datetime
    scores = result.to_pandas()

    lines = [
        "# RAG 知识库评估报告",
        f"\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"评估样本数：{len(rows)}",
        "\n## 总体指标\n",
        "| 指标 | 均值 | 说明 |",
        "|------|------|------|",
        f"| Faithfulness | {scores['faithfulness'].mean():.3f} | 回答忠于检索内容的程度（越高越好）|",
        f"| Answer Relevancy | {scores['answer_relevancy'].mean():.3f} | 回答与问题的相关程度（越高越好）|",
        f"| Context Recall | {scores['context_recall'].mean():.3f} | 检索内容覆盖标准答案的程度（越高越好）|",
        "\n## 逐题详情\n",
        "| # | 问题 | Faithfulness | Answer Relevancy | Context Recall | 来源 |",
        "|---|------|:---:|:---:|:---:|------|",
    ]

    for i, (row, qa) in enumerate(zip(rows, qa_pairs)):
        f = scores.iloc[i]["faithfulness"]
        ar = scores.iloc[i]["answer_relevancy"]
        cr = scores.iloc[i]["context_recall"]
        q = row["question"][:40] + ("..." if len(row["question"]) > 40 else "")
        lines.append(f"| {i+1} | {q} | {f:.2f} | {ar:.2f} | {cr:.2f} | {qa['source']} |")

    lines += [
        "\n## 问答详情\n",
    ]
    for i, row in enumerate(rows):
        lines += [
            f"### Q{i+1}: {row['question']}",
            f"\n**标准答案：** {qa_pairs[i]['ground_truth']}",
            f"\n**RAG 回答：** {row['answer']}",
            f"\n**检索到 {len(row['contexts'])} 条上下文**\n",
        ]

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ 报告已写入 {REPORT_FILE}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-gen", action="store_true", help="跳过问题生成，使用已有 questions.json")
    parser.add_argument("--top-k", type=int, default=4, help="检索条数")
    args = parser.parse_args()

    candidates = _get_candidates()
    client, model, _ = candidates[0]

    # 生成或加载问题
    if args.skip_gen and QUESTIONS_FILE.exists():
        print(f"📂 加载已有问题：{QUESTIONS_FILE}")
        qa_pairs = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    else:
        qa_pairs = generate_questions(client, model)
        QUESTIONS_FILE.write_text(json.dumps(qa_pairs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"💾 问题已保存到 {QUESTIONS_FILE}，可手动修改后用 --skip-gen 重新评估")

    if not qa_pairs:
        print("❌ 没有可用的问题，退出")
        return

    print(f"\n共 {len(qa_pairs)} 个问题")

    # 评估
    result, rows = evaluate(qa_pairs)

    # 写报告
    write_report(result, rows, qa_pairs)

    # 打印摘要
    scores = result.to_pandas()
    print("\n📈 评估结果摘要：")
    print(f"  Faithfulness:     {scores['faithfulness'].mean():.3f}")
    print(f"  Answer Relevancy: {scores['answer_relevancy'].mean():.3f}")
    print(f"  Context Recall:   {scores['context_recall'].mean():.3f}")


if __name__ == "__main__":
    main()
