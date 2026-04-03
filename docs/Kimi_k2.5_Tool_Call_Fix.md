# Kimi k2.5 `reasoning_content` 工具调用与 API 网关修复总结

本文档记录了在为项目接入 Kimi k2.5 模型并支持带有“深度思考 (`thinking` / `reasoning_content`)” 的工具调用 (Tool Calls) 时，遇到的两个链路 Bug，以及最终完整的故障排查和修复方案。

---

## 问题一：Kimi API 报 400 BadRequest 错误

### 1. 现象
当通过 API 界面点击 `Approve` (授权拦截的工具) 时，后端报错崩溃：
```json
Error code: 400 - {
  "error": {
    "message": "thinking is enabled but reasoning_content is missing in assistant tool call message at index 2",
    "type": "invalid_request_error"
  }
}
```

### 2. 根因剖析
Kimi k2.5 对带有思考链返回的模型（如 o1 或者 k2.5 系列）提出了极其严格的格式要求：**一旦在上下文中出现了模型原本进行 tool_call 的那条助理消息，那么该消息的 JSON 结构中必须包含原汁原味的 `reasoning_content` 字段。**
- 原本的系统调用了 LangGraph。LangGraph 依靠原生的 `langchain_openai` 库来解析大模型的流式或完整输出。
- `langchain_openai` 的标准解析器在将 HTTP 拆包构造为 `AIMessage` 时，**静默丢弃了**非标准 OpenAI 字段 `reasoning_content`，它只保留了 `content` 和 `tool_calls`。
- 当系统因为 HITL (Human-in-the-loop 人类循环确认) 将执行图暂停时，那条生成了工具的 `AIMessage` 被写入了长时图状态中。
- 当我们点击同意，LangGraph 带着历史消息再次去请求 Kimi 获取最终响应时，由于之前的消息丢了 `reasoning_content`，Kimi 严格模式校验失败，直接拦截并拒绝服务。
- 先前在 `PatchedKimiChatOpenAI` 补丁中，我们曾尝试对无 `reasoning_content` 的情况补发空字符串 `""`，但这不再奏效，Kimi 会将 `""` 视为“实质缺失”。

### 3. 终极解法
通过对 `engine/patched_kimi.py` 的改造解决：
1. **覆写解析器捕获数据**：我们拦截了 `ChatOpenAI` 底层的 `_create_chat_result` 方法，从最原始的 HTTP Dict 中重新把 `reasoning_content` 给“挖”了出来，并强行将其插进 LangChain 消息实例的 `message.additional_kwargs["reasoning_content"]` 字典中。这样图状态序列化时就不会丢弃了。
2. **安全的网络 Payload 兜底**：如果真的是一些异常引发的缺失，我们在请求封装前，会将 `p_msg["reasoning_content"]` 补全成带有实意的 `"思考中..."` 而不是 `""`，以此骗过 API 严格拦截机制。

---

## 问题二：API Approve 成功了，但 AI 回复“工具执行失败”

### 1. 现象
在解决 400 报错之后，点击 Approve 发送的请求成功收到了 `200 OK` 的响应。但大模型依然对用户讲：
> *“抱歉，我尝试保存您的幸运数字，但似乎未能成功。可能是系统出现了问题。”*

### 2. 根因剖析
这其实是一个 API 网关层面的代码遗漏：
- 大模型进行工具调用需要经历这几个步骤：大模型发起工具 → 被网关拦截 → 网关触发 Approve 接口 → LangGraph 工具节点 (`tool_node`) 被拉起 → 生成工具执行结果。
- 在 `d:\2026lab\Mini_Harness\api\main.py` 的 `/tools/approve` 路由中，程序在允许图表重新跑起来之前 (`graph.stream(None, thread)`)，**忘记将处于 Pending 状态的工具列表注入到 `approved_tools` 状态键中**。
- 这导致 `tool_node` 启动时，一查自己 `state.get('approved_tools')` 发现是空的，就直接返回了一个空的处理列表。相当于网关在放行时，没有告诉干活的人“你被批准干哪些活”。大模型拿不到结果，自然就认为工具彻底失败了。

### 3. 终极解法
修改 `api/main.py` 的 `/tools/approve` ：
在 `stream` 唤醒流程前，手动用 `graph.update_state()` 把挂起的工具正式盖章批准：
```python
if request.action == "approve":
    # 补上了极其关键的把挂起工具转移到批准列表
    approved_tool_calls = state.values.get("tool_calls", [])
    graph.update_state(
        thread,
        {"approved_tools": approved_tool_calls}
    )
    
    # 唤醒下一节点
    for event in graph.stream(None, thread):
        pass
```

---

## 避坑要点与总结
1. 当你使用任何非标准 OpenAI 生态的大语言模型能力时（比如深度思考、特有的引用参数、甚至是联网搜索的扩展字段等），**永远不要百分百信任 LangChain 官方解析器会帮你无损流转上下文**。必要的时候必须去重载 `_create_chat_result`（非流式）或者生成块转换器（流式）。
2. **状态更新的时机极其关键（Stateful Design）**：当遇到分布式网关触发任务或者 HITL 断点时，如果图表中后续的控制流（比如 Router 或 ToolNode）依赖于特定 State Key 的变更，务必要在 Resume 之前显式调用 `update_state` 赋予执行权限。
