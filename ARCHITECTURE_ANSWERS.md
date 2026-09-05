
## 模块一 / Q2

我不会按固定轮数反复总结整个 transcript，应该是把 active context 分成结构化 durable state、长期 summary 和 recent raw window。已经确认的目标、决定、pending item 与有状态工具结果进入 durable state；较老对话压成 summary；最近若干轮保留原文，以维持“第二个”“不是这个，是上一个”这类局部指代。压缩由 token budget 触发，并为输出与潜在工具结果预留 headroom；大工具输出只保留 compact observation 和可恢复 pointer。这样可以避免 recursive summary drift，重要 decision 不依赖多次自然语言总结。

## 模块二 / Q1

我认为 Memory 的核心不是存得更多，应该是判断当前回答是否真的需要历史信息。我会先加 recall gate，只有当问题依赖个人偏好、旧 decision、episodic history 或 task continuity 时才检索；普通事实问题直接回答。Memory 分为 semantic、episodic 和 task 三类，检索后按 relevance、confidence、freshness 与 expected answer gain rerank，并做 conflict/staleness check；当前明确约束始终高于旧 memory。最终只把少量高价值 memory 放进独立的 MEMORY 区域，避免无关历史占用 context 或干扰当前判断。

## 模块三 / Q2

我会把长期任务拆成 TaskDefinition 和不可变的 TaskRun。前者保存 instruction、schedule、timezone 和 delivery 方式；后者表示某个 local calendar day 的一次具体执行，并使用 `task_id + logical_date` 作为 idempotency key。每天 9 点由 scheduler 创建对应 run，按用户时区读取前一自然日的 session events，先抽取 topics、decisions、completed work 与 unresolved items，再生成 review。这样 worker 失败可以安全 retry，不会重复发送同一天的复盘，“昨天”也不会被错误解释成简单的 `now - 24h`。

## 模块四 / Q2

我的原则是 execution 可以并发，但同一 session 的 state mutation 必须串行化。每个 session 可以作为一个轻量 actor，用户消息、tool completion、failure、cancel 等事件都进入 mailbox，只有该 actor 能提交 state。每条 reasoning branch 带 `generation_id`：如果用户在洛杉矶天气查询尚未完成时改问北京，新消息可以 supersede 旧 generation；洛杉矶天气结果晚到时仍可记录，但不会自动进入当前 context。

对于有副作用的工具，还要区分 `cancel_requested`、`already_committed` 等状态，因为外部提交不能被假装回滚。我认为这样比单一的 `busy=True/False` 更能表达真实并发语义。

## 模块五 / Q1

Claude Code 的 tool use 对我来说更像一条持续的 action–observation 流，模型发出工具调用，Runtime 执行，再把结果作为结构化 observation 放回上下文，适合长任务、文件修改和复杂执行状态；OpenAI-compatible function calling 更像标准 RPC，模型主要负责给出函数名和参数，Runtime 执行后再把结果返回，协议更轻，也更容易和具体 provider 解耦。

如果让我自己设计 Harness，我会采用第二种思路，但内部再定义统一的 `ToolCall` 和 `ToolResult`，不让 Runtime 直接依赖 OpenAI 或 Claude 的消息格式。这样 provider 只是协议的翻译层，真正的参数校验、执行、错误处理、session、trace 和 context 都由 Harness 自己管理。我觉得这样边界很清晰，模型负责决定下一步做什么，Harness 负责保证这一步被可靠地执行，并把结果变成下一轮可用的状态。
