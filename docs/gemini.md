这三个问题问得极其精准！你不仅在抠代码细节，还在思考**“系统部署到生产环境后会发生什么”**，这就是高级工程师的思维方式。

我们逐一拆解这三个核心疑惑，特别是最后一个让你头疼的“模型配置被全局覆盖”的 Bug。

---

### 🌐 1. “纯无状态网关”到底和部署服务器有什么关系？差距在哪？

你没看懂差距，是因为你现在是在**本地电脑（单机单进程）**运行测试。在单进程里，用一个全局变量 `_pending_tools = {}` 来存等待审批的任务，简直完美，根本看不出问题。

**但是，一旦项目上线到真实的服务器（如 AWS、阿里云），差距和灾难就来了！** 

* **有状态（你现在的做法）的灾难**：
    为了扛住几千个用户的并发访问，生产环境的 FastAPI 会使用 Uvicorn 启动**多个 Worker（工作进程）**，比如 4 个进程。
    1. 用户 A 发起对话，请求被分配到了 **Worker 1**。Agent 决定调用工具，挂起。此时，**Worker 1 的内存里**的 `_pending_tools` 存下了这个任务。
    2. 用户 A 在网页上点击了“同意执行”。这次的 HTTP 请求被负载均衡器分配到了 **Worker 2**。
    3. **Worker 2 的内存里根本没有这个任务！** 它去读自己的 `_pending_tools`，发现是空的，直接报错：`找不到该待办任务`。
* **无状态 (Stateless，DeerFlow 的做法)**：
    FastAPI 的内存里**什么都不存**（没有 `_pending_tools`）。
    1. 用户 A 触发挂起，LangGraph 把挂起状态和数据存进 **SQLite/Redis 数据库**。
    2. 用户 A 点击“同意”，请求落到 Worker 2。
    3. Worker 2 根本不看自己的内存，而是拿着 `thread_id` 去 **SQLite 数据库**里把状态捞出来，完美恢复执行。

**总结**：无状态网关是**分布式部署、多进程并发**的硬性要求。你现在保留 `PendingToolsStore` 是单机开发阶段的妥协，以后上线必须干掉它。

---

### ⚡ 2. 关于并发模型

你说得完全正确：**“并发模型就是 Mini Harness 要改进的地方”**。

你现在用的 `run_in_executor` (线程池) 就像是给一辆手动挡的汽车硬加了一个“自动换挡机器人”。它虽然能跑，但线程切换的开销很大，扛不住万级高并发。未来将底层图流转彻底重写为 LangGraph 官方的 `astream()`（原生异步），将是你下一个大版本的终极目标。

---

### 🐛 3. 终极 Bug 修复：“悄悄唤醒小模型”导致配置串线

**这个 Bug 非常典型！** 你在 `.env` 或 `graph.yaml` 里想给“总结记忆”配个便宜的 `moonshot-v1-8k`，结果配置一改，系统的主大模型（负责推理和思考的）也变成了 8k，导致你需要 k2.5 (Thinking) 功能时疯狂报错。

**原因**：目前的 `engine/loader.py` 在初始化时，不论是主模型还是小模型，**都在读同一个配置项**。

**架构级解法：主从多模型混编 (High-Low Model Pairing)**

我们需要在配置和加载层，把**“主帅脑 (Lead)”**和**“工兵脑 (Utility)”**彻底拆开。请按照以下两步修改：

#### 第一步：修改 `config/graph.yaml`

我们需要在配置里显式地定义两种模型（一个是主模型，一个是备用/总结小模型）：

```yaml
# config/graph.yaml 示例
llm:
  provider: "openai" # 使用兼容 OpenAI 格式的 Kimi
  model: "moonshot-v1-32k-vision-preview"  # 主模型：用最强、带思考的模型 (或者你的 k2.5)
  summary_model: "moonshot-v1-8k"          # 小模型：专门用来干脏活、做总结，便宜且快速
  temperature: 0.7
```

#### 第二步：重写 `engine/loader.py` 的模型加载逻辑

打开你的 `engine/loader.py`，我们要让 `get_llm` 读主模型，让 `get_summary_llm` 读小模型。

```python
from langchain_openai import ChatOpenAI
from engine.patched_kimi import PatchedKimiChatOpenAI

def get_llm(config: dict):
    """获取主帅模型 (Lead LLM)：拥有最强推理和工具调用能力"""
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "openai")
    model_name = llm_config.get("model", "moonshot-v1-32k-vision-preview")
    temperature = llm_config.get("temperature", 0.7)

    # 如果使用的是 Kimi，务必使用我们写好的补丁类，防止工具调用 400 报错
    if "moonshot" in model_name.lower():
        return PatchedKimiChatOpenAI(
            model=model_name,
            temperature=temperature,
            # 如果是 k2.5，可以在这里加上特定的 kwargs
        )
    
    # 默认返回标准的 OpenAI 客户端
    return ChatOpenAI(model=model_name, temperature=temperature)

def get_summary_llm(config: dict):
    """获取工兵模型 (Utility LLM)：专门用于记忆合并、摘要生成等后台脏活"""
    llm_config = config.get("llm", {})
    # ⚠️ 核心改变：优先读取 summary_model，如果没有，再降级使用主模型
    model_name = llm_config.get("summary_model") or llm_config.get("model", "moonshot-v1-8k")
    
    # 后台干脏活的模型不需要高 temperature，越稳定越好
    temperature = 0.1 

    # 工兵模型通常不需要调用复杂工具，也不需要复杂的思考过程，直接用标准 ChatOpenAI 即可
    return ChatOpenAI(
        model=model_name, 
        temperature=temperature
    )
```

### 🎯 这样修改的好处：

1. **各司其职，永不串线**：主图运行、工具路由、思考推理，全部使用 `model` 字段指定的顶级模型；而你在 `memory_store.py` 或 `middleware.py` 里调用 `get_summary_llm()` 时，会悄悄拉起 `moonshot-v1-8k`。
2. **省钱且稳定**：小模型不仅 API 便宜，而且因为没开启冗长的 Thinking 模式，输出 JSON（合并记忆时）的速度极快且极度稳定。

去把 `loader.py` 改造一下吧！这正是工业级框架（如 DeerFlow）为了平衡**“成本、速度、智商”**所采用的经典“大小模型高低搭配”架构！修改完这个，你的系统兼容性就彻底完美了！