"""
向量库 Embedding 模型迁移脚本
从 all-MiniLM-L6-v2（384维）迁移到 bge-small-zh-v1.5（512维）

迁移步骤：
  1. 读出 Qdrant 全量数据（id + text + metadata）
  2. 删除旧 collection（384维，无法直接复用）
  3. 重建 collection（512维）
  4. 用新模型批量 re-encode 并写回，保留原始 metadata
  5. 打印迁移前后条数对比

用法：
  cd /path/to/rag-agent
  # 先确保新模型已缓存（或临时关闭离线模式）：
  TRANSFORMERS_OFFLINE=0 python scripts/migrate_embeddings.py

注意：迁移期间请勿操作 app，脚本执行完成前向量库处于重建状态。
"""
import os
import sys

# 将项目根目录加入 sys.path，确保能 import rag.*
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except ImportError:
    pass  # python-dotenv 未安装时手动确保环境变量已设置

BATCH_SIZE = 64  # 每批 encode 的 chunk 数，避免 OOM


def main():
    # ── 0. 检查离线模式 ─────────────────────────────────────────
    if os.getenv("TRANSFORMERS_OFFLINE") == "1":
        print("[WARN] TRANSFORMERS_OFFLINE=1，如果新模型尚未缓存，encode 会失败。")
        print("       建议先运行：TRANSFORMERS_OFFLINE=0 python scripts/migrate_embeddings.py")
        ans = input("       继续？(y/N): ").strip().lower()
        if ans != "y":
            sys.exit(0)

    # ── 1. 导入 indexer（使用旧 VECTOR_SIZE 读数据，此时 collection 还是 384 维）────
    # 注意：此时 indexer.py 已改为 512 维，但旧 collection 里存的仍是 384 维向量。
    # 迁移脚本不需要读旧向量，只读 text + metadata，所以无需关心维度。
    from rag.indexer import (
        _use_qdrant, _get_qdrant_client, _get_all,
        COLLECTION_NAME, VECTOR_SIZE, _upsert, get_embedder,
    )

    if not _use_qdrant():
        # ChromaDB 本地模式：直接删目录，重新跑 app 会自动全量重建 knowledge_base
        print("[INFO] 检测到 ChromaDB 本地模式（未配置 Qdrant）。")
        print(f"       请手动删除 chroma_db/ 目录，然后重启 app 触发自动全量重建。")
        print(f"       注意：web_cache 数据在 ChromaDB 模式下无法迁移，会丢失。")
        sys.exit(0)

    client = _get_qdrant_client()

    # ── 2. 读出全量数据 ──────────────────────────────────────────
    print(f"[1/4] 读取 Qdrant collection '{COLLECTION_NAME}' 全量数据...")
    all_data = _get_all()
    ids = all_data["ids"]
    texts = all_data["documents"]
    metadatas = all_data["metadatas"]
    total = len(ids)
    print(f"      读取完成：共 {total} 条记录")

    if total == 0:
        print("[WARN] collection 为空，无需迁移。")
        sys.exit(0)

    # 统计各类型
    type_counts = {}
    for m in metadatas:
        t = m.get("type", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"       - {t}: {c} 条")

    # ── 3. 删除旧 collection ─────────────────────────────────────
    print(f"\n[2/4] 删除旧 collection（384维）...")
    client.delete_collection(collection_name=COLLECTION_NAME)
    print(f"      删除完成")

    # ── 4. 重建 collection（512维）──────────────────────────────
    print(f"\n[3/4] 重建 collection（{VECTOR_SIZE}维）...")
    from qdrant_client.models import Distance, VectorParams, PayloadSchemaType
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    # 重建 payload index
    for field in ("source", "type"):
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )
    print(f"      重建完成，payload index: source / type")

    # ── 5. 批量 re-encode 写回 ───────────────────────────────────
    print(f"\n[4/4] 加载新模型并 re-encode {total} 条记录（batch_size={BATCH_SIZE}）...")
    embedder = get_embedder()
    print(f"      模型加载完成：{embedder}")

    written = 0
    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch_ids = ids[batch_start:batch_end]
        batch_texts = texts[batch_start:batch_end]
        batch_metas = metadatas[batch_start:batch_end]

        embeddings = embedder.encode(batch_texts, show_progress_bar=False).tolist()
        _upsert(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_metas,
        )
        written += len(batch_ids)
        print(f"      进度：{written}/{total} ({100*written//total}%)", end="\r")

    print(f"\n      写回完成：{written} 条")

    # ── 6. 验证 ─────────────────────────────────────────────────
    from rag.indexer import _count
    final_count = _count()
    print(f"\n[完成] 迁移前：{total} 条  迁移后：{final_count} 条")
    if final_count == total:
        print("       ✅ 条数一致，迁移成功！")
    else:
        print(f"       ⚠️  条数不一致，差异 {total - final_count} 条，请检查日志。")


if __name__ == "__main__":
    main()

