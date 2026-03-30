这是一份标准的\*\*高级工程师级别的“技术架构设计文档（Tech Spec / RFC）”\*\*雏形。

在面试时，如果你能按照“需求 -\> 选型 -\> 架构 -\> 落地”的逻辑向面试官清晰地推演整个项目，这本身就是极强的架构能力体现。

以下是为你量身定制的 **Mini Agent Harness（企业级智能体底座）** 的细化方案：

-----

# 🚀 Mini Agent Harness 架构设计与落地方案

## 一、 需求分析 (Requirements Analysis)

我们要做的不是一个面客的聊天机器人，而是一个\*\*“为生成 Agent 而生的引擎”\*\*。它的核心诉求如下：

1.  **业务解耦需求 (Configuration-Driven)**：系统代码中**绝对不能**硬编码任何特定的系统提示词或业务流。所有的“业务身份（如：代码审查员、行业分析师）”必须通过动态读取外部的 Markdown 文件来装载。
2.  **长链路并发需求 (Multi-Agent Routing)**：对于复杂任务，系统不能依赖单次 LLM 调用。需要具备“主节点拆解任务 -\> 拉起多个子节点并行处理 -\> 统一汇总”的编排能力。
3.  **安全风控需求 (HITL)**：在执行高危动作（如修改文件、发送数据）前，必须具备挂起（Suspend）整个系统进程的能力，等待人工审批后，再从断点精确恢复（Resume）。
4.  **上下文安全需求 (Context Engineering)**：系统必须能在无限轮次的对话中存活，具备精确计算 Token、并在接近阈值时自动触发“旧对话折叠/压缩”以及“高价值记忆动态注入”的能力。

-----

## 二、 技术选型与论证 (Technology Stack & Justification)

在面试中，“为什么选A不选B”比“用了A”更重要。

| 模块 | 技术选型 | 竞品对比与选择理由 (面试背书) |
| :--- | :--- | :--- |
| **核心编排引擎** | **LangGraph** | ❌ 弃用传统 LangChain / AutoGen：传统的 Chain 是线性的，无法实现复杂的循环（反思重试）；AutoGen 偏向多对话黑盒。<br>✅ 选择 LangGraph：它是基于图状态机（StateGraph）的，原生支持 Checkpointer，这是我们实现 **HITL（断点挂起与恢复）** 和子智能体并发的唯一最优解。 |
| **API 与通信层** | **FastAPI** | ❌ 弃用 Flask / Django：并发性能弱，原生不支持异步。<br>✅ 选择 FastAPI：原生全异步（`async/await`），完美支持多智能体高并发，且极其容易实现 **SSE (Server-Sent Events)** 协议，用于向前端流式吐出 Agent 的“思考过程”。 |
| **状态与记忆存储** | **SQLite + JSON** | ❌ 弃用 Redis / MySQL：对于一个开源 Mini 脚手架来说太重了，增加部署成本。<br>✅ 选择 SQLite：轻量级本地文件数据库，足以支撑 LangGraph 的持久化状态存储（Checkpointer）和用户的长期事实记忆（Facts Memory）存取。 |
| **Token 预算控制** | **tiktoken** | ✅ 这是 OpenAI 官方的 Tokenizer。在每次请求发给 LLM 前，用它在本地做极其精准的长度计算，从而决定是直接发请求，还是先触发“摘要压缩拦截器”。 |
| **可观测性监控** | **LangSmith** | ✅ Agent 开发的刚需。代码里的几个环境变量一开，图节点流转、耗时、甚至并发的子智能体内部状态，全都在云端仪表盘可视化，降维打击普通开发者的 `print()` 调试。 |

-----

## 三、 核心架构设计 (Architecture Design)

整个脚手架可以分为四个核心子系统（这也是我们后续写代码的四个文件夹）：

### 1\. `Engine` (执行引擎层)

  * **状态定义 (`ThreadState`)**：继承自 LangGraph 的基础状态，扩展出 `task_list`（子任务列表）、`approved_tools`（已授权工具）等字段。
  * **节点图 (`Graph`)**：包含 `LeadAgent_Node`（主节点，负责规划）、`SubAgent_Node`（子节点并发）、`Tool_Execution_Node`（工具执行）和 `Human_Approval_Node`（人工审批拦截器）。

### 2\. `Context & Memory` (上下文与记忆层 / 中间件)

  * 采用**拦截器（Middleware）设计模式**。
  * 在 `LeadAgent_Node` 真正调用大模型前，流经两个拦截器：
      * *Token Budget Middleware*：用 `tiktoken` 算一下。超过 8000 tokens？触发小模型把前文总结成 500 字摘要。
      * *Memory Injection Middleware*：去 SQLite 查出该用户的历史习惯（如：“输出请用中文”），插入到 System Prompt 的 `<memory>` 标签中。

### 3\. `Skills & Tools` (能力挂载层)

  * 实现一个轻量级的 `SkillLoader` 类。
  * 功能：程序启动或切换场景时，读取 `/skills/research.md`，解析其中的 Frontmatter（YAML 头）获取所需工具列表，并将正文直接注入为 Lead Agent 的人设。

### 4\. `Gateway` (网关接入层)

  * FastAPI 路由。负责接收 HTTP 请求，初始化 LangGraph，并将 LangGraph 内部的事件流转（`yield` 出的事件）转换为 SSE 格式发送给客户端。

-----

## 四、 落地方案与执行路径 (Execution Path)

为了避免陷入代码泥潭，我们要分三个阶段（Phase）来敏捷开发这个项目。这也是我们接下来带你写代码的顺序：

### 阶段一：搭建“大脑与刹车”（Core Graph & HITL） 👈 **我们先做这个**

  * **目标**：跑通基于 LangGraph 的主循环，并实现最核心的“断点审批”风控机制。
  * **动作**：
    1.  定义 `ThreadState`。
    2.  创建最简单的 Agent 节点和 Tool 节点。
    3.  引入 `MemorySaver`（LangGraph 内置的内存 Checkpointer）。
    4.  **高光时刻**：在图流转到 Tool 节点前，设置 `interrupt_before=["tools"]`。演示代码跑一半挂起，输入 `y` 继续执行的效果。

### 阶段二：建立“护城河”（Memory Injection & Token Middleware）

  * **目标**：解决长对话崩溃问题。
  * **动作**：写一个简单的伪数据库逻辑，实现 `tiktoken` 预算计算和动态组装 System Prompt 的逻辑，在调用节点前做拦截修改。

### 阶段三：包上“工程外壳”（Skill Parsing & FastAPI SSE）

  * **目标**：让它从一个 Python 脚本变成一个真正的后台服务。
  * **动作**：写 Markdown 解析器，用 FastAPI 包装整个 Graph，实现接口调用。

-----
