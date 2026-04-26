# Design: 工具结构化错误响应

## 现状分析

`agent/tools.py:62-83`，`search_knowledge_base` 空结果时：
```python
return {"results": [], "message": "知识库中未找到相关内容，建议调用 search_web 继续搜索"}
```

问题：`message` 字段语义模糊，Agent 依赖 LLM 理解文本才能触发降级，不够结构化。

## 方案对比

| 方案 | 核心思路 | 优点 | 缺点 | 适用场景 |
|------|----------|------|------|----------|
| A：新增 `suggestion` 字段 | 在现有响应中加 `suggestion: "建议调用 search_web 继续检索"` | 改动最小，向后兼容，`message` 保留 | 字段略冗余 | 本项目（改动最小） |
| B：重命名 `message` → `suggestion` | 直接替换字段名 | 语义更清晰 | 需同步更新所有 mock 和断言 | 重构场景 |
| C：引入错误码 | `{"results": [], "error_code": "KB_EMPTY", "suggestion": "..."}` | 机器可读，扩展性强 | 过度设计，当前只有一种错误 | 多工具、多错误类型场景 |

**推荐：方案 A**
理由：改动范围最小（只加一个字段），不破坏现有测试中已有的 `message` 断言，符合 YAGNI 原则。

## 技术设计

### 涉及文件

- `agent/tools.py:71-73` — 空结果返回值加 `suggestion` 字段
- `agent/tools.py:66-69` — 异常返回值加 `suggestion` 字段（一致性）
- `tests/test_acceptance.py:204` — 同步 mock 返回值（加 `suggestion`）
- `tests/test_acceptance.py` — 新增 `test_kb_empty_suggestion_field` 测试

### 改动细节

**tools.py 改动**（2 处）：
```python
# 空结果（line 71-73）
if not chunks:
    return {
        "results": [],
        "message": "知识库中未找到相关内容，建议调用 search_web 继续搜索",
        "suggestion": "建议调用 search_web 继续检索",   # 新增
    }

# 异常（line 66-69）
return {
    "results": [],
    "message": "知识库检索异常，建议调用 search_web 继续搜索",
    "suggestion": "建议调用 search_web 继续检索",   # 新增
}
```

**新增测试**（tests/test_acceptance.py）：
```python
class TestToolStructuredError:
    @patch("agent.tools.retrieve")
    def test_kb_empty_suggestion_field(self, mock_retrieve):
        mock_retrieve.return_value = []
        result = search_knowledge_base("不存在的内容")
        assert result["results"] == []
        assert "suggestion" in result
        assert "search_web" in result["suggestion"]
```

## 决策记录

- [x] 已确定：新增 `suggestion` 字段而非重命名 `message` — 向后兼容，改动最小
- [x] 已确定：异常路径也同步加 `suggestion` — 保持两个空结果路径语义一致
