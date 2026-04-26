# Tasks: 工具结构化错误响应

## 实现清单

- [ ] 1. 在 `agent/tools.py` 空结果和异常返回值中加 `suggestion` 字段
  - 文件：`agent/tools.py:66-73`
  - 改动：两处 `return` 各加一行 `"suggestion": "建议调用 search_web 继续检索"`
  - 验收：改动后 `search_knowledge_base("不存在内容")` 返回值含 `suggestion` 字段

- [ ] 2. 同步 `TestKbEmptyFallback` 中的 mock 返回值
  - 文件：`tests/test_acceptance.py:204`
  - 改动：mock 的 `return_value` 加 `"suggestion"` 字段，与实际代码保持一致

- [ ] 3. 新增 `TestToolStructuredError` 测试类
  - 文件：`tests/test_acceptance.py`（在 `TestKbEmptyFallback` 后新增）
  - 内容：mock `retrieve` 返回空列表，直接调用 `search_knowledge_base`，断言 `suggestion` 字段存在且含 `"search_web"`

- [ ] 4. 运行验收测试，确认全部通过
  - 命令：`cd /Users/zzy/others/project11/rag-agent && venv/bin/pytest tests/test_acceptance.py -v`

- [ ] 5. 更新 `design/agent_optimization.md` TODO #10 状态为 ✅ 已完成，追加变更记录

## 完成标准

所有 task 打勾 + AC-1 至 AC-4 全部通过 + 验收测试绿色
