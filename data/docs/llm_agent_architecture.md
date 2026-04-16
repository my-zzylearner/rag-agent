---
topic: LLM Agent架构
keywords: llm agent, function calling, react, multi-agent, planning, tool use, autogen, crewai, langgraph, orchestration
description: 介绍LLM Agent的核心架构（规划/工具/记忆/执行循环）、Function Calling机制、ReAct推理框架及Multi-Agent协作模式，适用于判断与智能体设计、工具调用、Agent框架选型相关的知识路由。
type: knowledge_base
---

# LLM Agent 架构与工具调用详解

## 什么是 LLM Agent

LLM Agent 是以大语言模型为核心推理引擎，能够自主规划、调用外部工具、循环执行直到完成任务的系统。

与普通 LLM 问答的区别：
- **普通 LLM**：输入 → 输出，一次性生成
- **LLM Agent**：输入 → 思考 → 行动 → 观察 → 思考 → ... → 输出，循环迭代

## Agent 核心组件

### 1. 规划（Planning）
LLM 负责分解任务、制定步骤、决定调用哪个工具。

常见规划范式：
- **ReAct**（Reasoning + Acting）：交替进行推理和行动，每步先思考再执行
- **Chain-of-Thought**：先生成完整推理链，再执行
- **Tree of Thoughts**：探索多条推理路径，选最优

### 2. 工具（Tools）
Agent 可以调用的外部能力，扩展 LLM 的边界：
- 搜索引擎（获取实时信息）
- 代码执行器（运行 Python/JS）
- 数据库查询
- API 调用（天气、股票、日历）
- 文件读写

### 3. 记忆（Memory）
- **短期记忆**：对话历史，存在 context window 中
- **长期记忆**：向量数据库，跨对话持久化
- **工具调用历史**：本轮任务中已执行的操作

### 4. 执行循环
```
用户输入
    ↓
LLM 推理（是否需要工具？用哪个？参数是什么？）
    ↓
[有工具调用] → 执行工具 → 将结果加入上下文 → 返回 LLM 推理
[无工具调用] → 生成最终回答 → 结束
```

## Function Calling（工具调用）机制

Function Calling 是现代 LLM 支持 Agent 的核心能力，让模型能以结构化方式调用外部函数。

### 工作流程

**Step 1：定义工具**
```json
{
  "name": "search_web",
  "description": "搜索互联网获取最新信息",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "搜索关键词"}
    },
    "required": ["query"]
  }
}
```

**Step 2：LLM 决策**
模型分析用户问题，判断是否需要调用工具，输出结构化的调用指令：
```json
{"name": "search_web", "arguments": {"query": "2024年AI搜索最新进展"}}
```

**Step 3：执行工具**
应用层接收调用指令，执行实际函数，返回结果。

**Step 4：结果回传**
将工具结果加入对话历史，模型基于结果生成最终回答。

### 支持 Function Calling 的模型
- GPT-4 / GPT-3.5-turbo（OpenAI）
- Claude 3（Anthropic）
- ERNIE 4.0 / ERNIE-Speed（百度文心）
- Gemini Pro（Google）
- Qwen（阿里通义）

## ReAct 框架详解

ReAct（Reasoning + Acting）是最经典的 Agent 范式，由 Google 2022 年提出。

### 格式
```
Thought: 我需要查找最新的 RAG 技术进展
Action: search_web
Action Input: {"query": "RAG技术2024最新进展"}
Observation: [搜索结果...]

Thought: 根据搜索结果，我可以回答用户的问题了
Final Answer: ...
```

### 优点
- 推理过程透明，可调试
- 每步都有观察反馈，能纠错
- 适合多步骤复杂任务

### 缺点
- token 消耗多
- 可能陷入循环
- 需要设置最大步数防止死循环

## Multi-Agent 系统

多个 Agent 协作完成复杂任务：

### 常见模式
- **主从模式**：Orchestrator Agent 分解任务，Sub-Agent 执行子任务
- **流水线模式**：Agent A 的输出作为 Agent B 的输入
- **辩论模式**：多个 Agent 从不同角度分析，最终投票或综合

### 代表框架
- **AutoGen**（Microsoft）：多 Agent 对话框架
- **CrewAI**：角色化 Multi-Agent
- **LangGraph**：基于图的 Agent 工作流

## RAG + Agent 结合

将 RAG 作为 Agent 的一个工具，是目前最实用的 AI 应用架构：

```
用户问题
    ↓
Agent 判断
    ├── 知识库有答案？ → 调用 RAG 工具检索
    ├── 需要实时信息？ → 调用 Web Search 工具
    ├── 需要计算？    → 调用 Code Executor 工具
    └── 直接回答？    → LLM 生成
    ↓
融合多路结果 → 生成回答
```

### 优势
- RAG 提供私域知识，Agent 提供动态能力
- 可以根据问题类型自动路由到最合适的信息源
- 比纯 RAG 更灵活，比纯 Agent 更稳定

## Agent 工程实践

### 防止死循环
```python
MAX_ITERATIONS = 5
for i in range(MAX_ITERATIONS):
    response = llm.call(messages, tools)
    if not response.tool_calls:
        break  # 无工具调用，任务完成
    # 执行工具...
```

### 错误处理
工具调用失败时，将错误信息返回给 LLM，让模型自行决定是否重试或换策略。

### 工具描述的重要性
工具的 description 字段直接影响模型的工具选择决策，应该：
- 清晰描述工具的用途和适用场景
- 说明输入输出格式
- 给出使用示例（可选）

### 成本控制
- 设置最大工具调用次数
- 使用轻量模型做工具选择，重量模型做最终生成
- 缓存频繁调用的工具结果

## 评估 Agent 效果

- **任务完成率**：Agent 能否正确完成任务
- **工具调用准确率**：选对工具、参数正确的比例
- **步骤效率**：完成任务所需的平均工具调用次数
- **幻觉率**：生成内容与工具结果不符的比例


---
## 补充知识（来自网络搜索）
> query: AI Agent框架对比 LangChain AutoGen CrewAI MetaGPT优缺点 | 更新时间: 2026-04-16 | hash: 6e87f54b68cea8866399c16471c50312

# AI Agent 框架技术对比

### 1. LangChain
*   **定位**：单智能体编排与链式调用基础框架。
*   **优点**：生态最完善，组件丰富（Memory, Tools），兼容性强，适合构建复杂的单一任务链。
*   **缺点**：学习曲线陡峭，链式结构复杂时调试困难，原生多智能体协作支持较弱（通常需结合 LangGraph 实现）。

### 2. AutoGen
*   **定位**：微软开源的多智能体对话框架。
*   **优点**：支持智能体间自由对话与协作，人机协同（Human-in-the-loop）机制成熟，适合研究复杂交互逻辑。
*   **缺点**：配置相对繁琐，对话轮次不可控易导致死循环或资源消耗过高，缺乏结构化流程控制。

### 3. CrewAI
*   **定位**：基于角色的多智能体协作框架。
*   **优点**：抽象层级高，易于上手（基于角色、任务、工具定义），支持顺序或层级流程，适合业务流程自动化。
*   **缺点**：底层灵活性不如 AutoGen，定制化深度受限，社区生态相对较新。

### 4. MetaGPT
*   **定位**：多智能体软件公司模拟框架。
*   **优点**：内置 SOP（标准作业程序），输出结构化强（如自动生成 PRD、代码），专精于软件开发场景。
*   **缺点**：场景特定性强，通用性较弱，非开发类任务适用性低。

### 选型建议
*   **快速开发/单任务**：首选 LangChain。
*   **复杂研究/自由对话**：选择 AutoGen。
*   **业务流程/角色分工**：推荐 CrewAI。
*   **自动化软件开发**：首选 MetaGPT。

---
**来源：**
*   大模型应用开发必备：AI Agent框架技术深度剖析与实战选择
*   到底选谁？五大多智能体( Multi-AI Agent) 框架对比
*   CrewAI、AutoGen、Dify、MetaGPT、OmAgent深度横评 - 知乎专栏
*   Agent调研--19类Agent框架对比（下） - 阿里云开发者社区


---
## 补充知识（来自网络搜索）
> query: LangChain LangGraph AutoGen CrewAI MetaGPT AutoGPT Agent框架优缺点对比选择 | 更新时间: 2026-04-16 | hash: 166bebc3ecd891852ea428348e45c849

# Agent 框架技术对比与选型

### 1. 框架核心特性
*   **LangChain**：基础编排框架。提供模型 I/O、检索、工具调用等组件，适合构建线性链，但在复杂循环和状态管理上较弱。
*   **LangGraph**：LangChain 的扩展，引入**循环图**与**状态机**机制。支持持久化和流式传输，适合构建可控、有状态的复杂多 Agent 应用。
*   **AutoGen**：微软推出的多 Agent 对话框架。核心是**对话模式**，支持 Agent 间自主通信与人机协同，适合研究复杂交互。
*   **CrewAI**：角色扮演型框架。以“角色-任务-小组”为抽象层，封装度高，代码量少，适合快速构建流程化协作系统。
*   **MetaGPT**：软件公司模拟框架。内置 **SOP（标准作业程序）**，将 Agent 映射为产品经理、架构师等角色，专注于自动化软件开发。
*   **AutoGPT**：早期探索性框架。完全自主驱动，试图通过循环实现目标，但易陷入死循环，缺乏结构化控制，生产环境可用性低。

### 2. 优缺点对比
*   **LangChain/LangGraph**：生态最完善，灵活性极高。缺点是抽象层复杂，学习曲线陡峭。
*   **AutoGen**：交互能力强，支持复杂对话流。缺点是难以预测运行走向，调试成本高，资源消耗较大。
*   **CrewAI**：上手快，开发效率高。缺点是底层控制力不足，定制化灵活性不如 LangGraph。
*   **MetaGPT**：软件工程领域效果好，输出规范。缺点是应用场景单一，通用性受限。

### 3. 选型建议
*   **复杂/生产级工作流**：首选 **LangGraph**（控制力强）。
*   **快速原型/业务流程自动化**：推荐 **CrewAI**。
*   **自动化软件开发**：推荐 **MetaGPT**。
*   **多 Agent 交互研究**：推荐 **AutoGen**。

### 来源
*   [多agent框架对比- 53AI-AI知识库](https://www.53ai.com)
*   [Agent 框架横向对比：LangChain、LangGraph、AutoGen - 知乎专栏](https://zhuanlan.zhihu.com)
*   [Agent 开发框架对比 - GitHub](https://github.com)
*   [2025年AI Agent Framework选型指南：从技术对比到生产实践](https://www.example.com)
