# Feature: 工具结构化错误响应

## 背景与目标

`search_knowledge_base` 空结果时已返回 `message` 字段（`agent/tools.py:72`），但：
1. 字段语义不明确（`message` 混用于正常信息和错误建议）
2. 无专项验收测试验证 Agent 在收到该响应后确实调用了 `search_web`
3. `TestKbEmptyFallback` 中 mock 的返回值硬编码了 `message` 内容，与实际代码不同步

目标：明确工具空结果响应的语义，补充专项验收测试，确保 Agent 降级行为可持续验证。

## 用户故事

- 作为用户，我希望当本地知识库没有答案时，Agent 能自动联网搜索，而不是直接说"未找到"，以便每次都能得到有实质内容的回答。

## 验收标准

- [ ] AC-1：`search_knowledge_base` 空结果时，响应包含 `suggestion: "建议调用 search_web 继续检索"` 字段
- [ ] AC-2：`TestKbEmptyFallback.test_kb_empty_falls_back_to_web` 中的 mock 与实际 `tools.py` 返回格式保持一致
- [ ] AC-3：新增测试 `test_kb_empty_suggestion_field`，直接调用 `search_knowledge_base`（mock `retrieve` 返回空列表），断言响应中包含 `suggestion` 字段
- [ ] AC-4：不影响知识库有结果时的正常流程（无 `suggestion` 字段）

## 约束与边界

- 只修改 `agent/tools.py` 和 `tests/test_acceptance.py`，不修改系统 prompt
- 不处理 `search_web` 也为空的情况（超出本 spec 范围）
- `message` 字段可保留（向后兼容），`suggestion` 作为新增字段
