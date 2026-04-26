# 方案设计文档

| 文件 | 主题 | 状态 | 最后更新 |
|------|------|------|----------|
| [agent_optimization.md](agent_optimization.md) | Agent 停止控制、联网搜索路由、知识自动内化、多模型 fallback、日志 trace | 进行中 | 2026-04-12 |
| [evaluation.md](evaluation.md) | RAG 评估体系、RAGAS 指标说明、历史评估结果 | 进行中 | 2026-04-12 |

## 新功能开发流程

新功能请使用 spec-first 工作流，在 [`.claude/specs/`](../.claude/specs/README.md) 下创建对应目录，按 `requirements.md → design.md → tasks.md` 顺序填写后再开始编码。

| Spec | 功能 | 状态 |
|------|------|------|
| [tool-structured-error](../.claude/specs/tool-structured-error/) | 工具结构化错误响应（KB 空结果自动降级） | 进行中 |
