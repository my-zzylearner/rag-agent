# Spec 工作流说明

本目录采用 Kiro 风格的 spec-first 开发模式，每个新功能在编码前先完成三个文档。

## 目录结构

```
.claude/specs/
└── <feature-name>/
    ├── requirements.md   # WHAT：用户故事 + 验收标准
    ├── design.md         # HOW：技术方案 + 架构决策
    └── tasks.md          # STEPS：实现清单（逐条打勾）
```

## 工作流程

```
1. requirements.md  →  明确"做什么、为什么"
        ↓
2. design.md        →  明确"怎么做"（含方案对比）
        ↓
3. tasks.md         →  拆解"做哪些步骤"（每条 = 一次 AI 对话）
        ↓
4. 逐 task 实现，完成后打勾
        ↓
5. 功能完成 → 同步关键决策到 design/agent_optimization.md
```

## 与 design/ 的关系

| | `design/`（现有） | `.claude/specs/`（本目录） |
|--|--|--|
| 用途 | 记录已完成的决策与变更历史 | 规划待实现功能的开发过程 |
| 时态 | 过去时（已做的） | 未来时（要做的） |
| 粒度 | 模块级 | 单功能级 |

## 当前 Specs

| 功能 | 状态 | 对应 design/ TODO |
|------|------|------------------|
| [web-cache-aging](web-cache-aging/) | 进行中 | agent_optimization.md #17 |
