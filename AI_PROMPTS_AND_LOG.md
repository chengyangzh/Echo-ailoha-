# AI Prompt + Problem-Solving Log

下面是对 Echo 开发过程中几次关键 AI 协作的整理版记录。

## 1. 先把 Echo 定义成一个有产品身份的 agent

**Prompt（整理）**：我不想做一个 calculator、search、todo 拼在一起的通用 agent demo，我希望它本身有一个明确而且有趣的用途。我想到的是 analogical reasoning：用户给出一个真实场景、失败、策略或者冲突，系统不要只按关键词找相似事件，而是先抽象这个场景里真正起作用的结构，再去找其他领域里因果关系相似的案例。这个抽象至少要能表达 roles、goal、strategy、mechanisms、turning point 和 outcome，而且我不希望把流程写死成 search→read→final，工具调用顺序还是应该由模型自己决定。搜索只能给候选，真正采用一个 analogy 之前必须再读具体 case；最后的回答除了说明 mapping 和 shared mechanism，也一定要说清楚 where the analogy breaks。

**处理结果**：最终把 Echo 定义成了一个“search by structure, not by words”的结构类比 agent。Runtime 只规定合法动作和少数不变量，LLM 自己选择 direct answer、`search`、`read_case`、`analogy_board` 或其他工具；新的 concrete situation 默认被解释成 implicit analogy request，这样 Echo 不会退化成普通聊天助手。这个决定也确定了后面所有功能的优先级：复杂度应该集中在结构抽象、候选检索、证据检查和跨轮状态，而不是泛化成一个无边界的 agent 平台。

## 2. 设计 Core Atlas 和 Wikipedia 的边界

**Prompt（整理）**：Echo 最大的问题是很容易嘴上说“结构相似”，实际还是 semantic similarity，所以我需要一个能真正区分 structure 和 surface 的 search space。我的想法是先做一个小而受控的 Core Atlas，把寓言、中外历史、战略、商业、制度、科学技术等不同领域的案例统一成同一种 CaseCard，这样同一个 mechanism 可以跨领域比较；但 Atlas 又不能变成 Echo 的全部世界知识，所以我还想留一个开放搜索出口。我不想加 vector database、embedding service 或 browser agent，那些会把 scope 拉得太大，而且也不直接证明 analogy 做得更好。请按这个思路设计一个最小的两层检索：本地 atlas 用来稳定搜索和评测，外部只接一个 source，先发现 candidate，再单独读取 bounded evidence，外部搜索层本身绝对不能替 LLM 做 analogy judgment。

**处理结果**：最后形成了 Core Atlas + Wikipedia 的两层结构。Core Atlas 提供可复现的结构化候选和 deterministic evaluation；Wikipedia 只通过 MediaWiki API 返回 title、snippet、page id、URL 等轻量候选，`read_case` 再读取有限长度的 source extract。Wikipedia 页面不会自动被转换成 CaseCard，也不会因为搜索命中就被视为好 analogy。后来又补了一层很小的 coverage gate：Core 第一名的 structural score 低于 0.30，或已经识别出的 canonical mechanism 在第一名里完全没有命中时，Runtime 不允许直接结束，而要求模型至少尝试一次 Wikipedia；但 query、候选选择和最终 structural judgment 仍然由 LLM 决定。这样 retrieval layer 负责“有哪些东西值得看”，LLM 负责“这些东西在关系结构上到底是不是同一个问题”，既扩大了 coverage，又没有把项目变成普通 RAG。

## 3. 把 transcript、长期摘要、证据和 analogy decision 分开

**Prompt（整理）**：我希望 session 真的能恢复，而且长对话不能靠把所有 raw history 永远塞回模型里来维持。我觉得至少要区分三件事：最近几轮原话，因为 follow-up 可能依赖指代；更早历史的压缩摘要，因为上下文不能无限增长；以及已经做出的结构化 analogy decision，例如哪个 candidate 被选中、哪个被拒绝、mapping 和 analogy break 是什么，这些不应该被一个 lossy summary 决定是否还能记住。外部 evidence 也类似，当前下一次 reasoning 可以需要比较完整的页面内容，但 SQLite 没必要永久保存整段 Wikipedia。请设计一个尽量小的 state model，让我可以明确解释每次 recall 发生在什么时候、什么内容会重新进入 context，以及 process restart 以后为什么还能续上。

**处理结果**：session 用 `(user_id, session_id)` 作为 SQLite key；每次模型决策前，ContextManager 按“active Analogy Board → session summary → recent raw messages/tool observations → latest rich tool result”的顺序重建 context。旧 transcript 超过预算时才压缩，最近窗口继续保留。Analogy Board 独立保存 selected/rejected candidate 和结构化 mapping，所以 transcript 被压缩也不会丢掉已经确认的决定。外部页面则采用“当前 rich、长期 compact”的策略：下一步判断可以看到 bounded evidence，持久化历史只保留更短的 provenance/evidence preview。这个设计最后被概括成一句话：the transcript remembers what was said; the board remembers what was decided。

## 4. 设计 surface trap

**Prompt（整理）**：我不想拿几个 demo 看模型回答得像不像就说 Echo 有 analogical reasoning。我希望评测本身能主动杀掉“其实只是关键词匹配”这个解释。可以借鉴 Hongjing Lu、Keith Holyoak 和 Taylor Webb 关于 LLM analogy 的实验思路，把 surface similarity 和 relational structure 分开：near analogy 是 surface 高、structure 高；far analogy 是 surface 低、structure 高；surface trap 是 surface 高但关键 causal relation 错了；另外再有 unrelated control。最重要的比较应该是 far structural match 能不能压过 surface trap。gold 也不能看到模型输出以后再挑，应该先冻结 mechanism 和 causal edge，再构造对应项和干扰项。另外加一个非常简单的 lexical baseline，如果 baseline 也能轻松做对，那说明 benchmark 本身不够难；还要把“raw situation 有没有抽象出正确 mechanism”和“给定 frame 后 retrieval 能不能找到正确 case”拆开测。

**处理结果**：evaluation 最后按 Near / Far / Surface Trap / Unrelated 四种条件组织，核心指标是 structural-over-surface accuracy，并同时看 Recall@5、MRR 和 lexical baseline。第一次 surface-trap benchmark 实际被否掉过，因为 lexical Jaccard baseline 做到了 93.75%，说明干扰项太弱；后来改成破坏关键 causal edge 而不是只换词，lexical baseline 明显下降，而 structural ranker 仍能稳定把 far match 排在 surface trap 前面。另一方面又单独做了 frame-extraction control，因为 retrieval 100% 并不能证明模型从原始用户输入里抽象对了机制。这个 separation 让 runtime correctness、abstraction quality 和 retrieval quality 不再混成一个指标。

## 5. Live bug：function schema 在 Runtime 之前就把合理 tool call 拒绝了

**问题记录**：真实 Groq 测试里，模型对一个 cybersecurity alert-fatigue 场景已经抓到了核心 mechanism，但它生成的 search 参数里出现了 `medicine`、`engineering` 这类开放 domain label，后来还有一次只给出 partial frame：`roles=[]`、`turning_point=""`、没有 `outcome`。这些 action 在语义上都可以继续搜索，但 provider 会先按 JSON Schema 验证 function call，于是请求直接 400，call 根本到不了 Runtime，也就谈不上本地 validation 或 error observation。这里暴露出来的问题不是“模型不会按 schema”，而是我们把 provider-facing action contract 和 evaluation 里的完整 AnalogyFrame 错当成了同一个东西。

**处理结果**：之后把两层 schema 拆开。provider-facing `search` 只严格约束数据形状和真正不可缺少的执行条件，允许 partial working frame，也允许开放语义标签；进入 tool 以后再把能识别的 mechanism/domain 映射到 Core Atlas 的有限 vocabulary，未知值不会导致整个请求失败。与此同时，controlled evaluation 仍然使用严格完整的 AnalogyFrame。像 `read_case.case_id` 和 calculator expression 这种没有就无法执行的字段继续保持 required。两次真实失败生成的参数形状都被直接做成 regression test，避免以后为了“schema 看起来严格”再次把合理的 agent action 挡在 Runtime 外面。

## 6. Live bug：provenance guard 造成 no-progress loop，以及 Echo 一度忘了自己是 Echo

**问题记录**：一次真实运行里，模型先 `search`，随后想直接给 final，但我们要求候选必须经过 `read_case`，于是 provenance guard 把 final 挡了回去；问题是它没有真正推进，而是稍微改写 query 后再次 search，候选集合完全没变，然后又被 guard 挡住。这个循环最终把同一 turn 的上下文撑大到 provider request limit。另一次测试暴露的是产品层问题：我只输入“我和多个女生暧昧，结果她们发现彼此以后都拒绝和我恋爱”，没有显式说 analogy，Echo 就直接变成普通 relationship assistant，输出一串坦诚沟通、反思成长之类建议，完全丢掉了这个产品最有意思的能力。

**处理结果**：no-progress 的判断不再看 query wording，而是看新一次 search 有没有产生 unseen candidate IDs；如果候选集合没有新增，Runtime 返回 `search_saturated`，让模型读取已有 case、真正修改 causal frame，或者在有理由时扩大 source。provenance guard 仍然不会替模型自动选 candidate，它只明确指出哪些候选尚未 inspect，从而保留 LLM 的 tool autonomy。产品 contract 也同时收紧：任何新的 concrete situation、dilemma、failure、strategy 或 surprising outcome 都默认是 implicit analogy request，即使用户没有说“类比”；generic advice 不再是第一反应，如果用户真的问怎么办，也应该先通过 analogy 找到 shared mechanism，再从机制推导 takeaway。这个中文 relationship 场景后来直接进入 live routing test，要求 trace 中实际出现 `search` 和 `read_case`。

## 7. Live quality failure：必须允许“拒绝检索结果并重新构造”

**问题记录**：submit-safe 版本解决了最危险的稳定性问题——同一个 turn 不会再因为 8 次 tool loop 耗尽而只返回 iteration-limit message。但真实测试暴露出更深的产品问题：对“朋友背叛并抢走工作名额”这个场景，Echo 最终给出了 Trojan Horse。这个答案表面上完整，实际上为了让故事成立，偷偷补进了“朋友长期伪装忠诚、先骗取信任、再趁机渗透”这些用户从未提供的前提。与此同时，trace 显示模型在 Core 和 Wikipedia 之间多次改写近义 query，说明它仍然把“找到一个有名字的 case”当成目标，而不是“找到一个真正解释得通的结构”。我希望 Echo 能产生类似“血液循环 ↔ 城市地铁网络”这种并不依赖知名故事、但在 flow、hub、capacity、routing、bottleneck 等关系上非常干净的类比。

**处理结果**：这一轮把 Echo 从单一路径 retrieval agent 改成 retrieval + construction 双通道。Core Atlas 只作为 seed set：coverage threshold 从 0.30 提高到 0.48，并额外检查 canonical mechanism coverage；如果出现 Atlas 无法表达的 open causal mechanism，也直接视为 weak coverage，要求至少尝试一次 mechanism-oriented Wikipedia discovery。生产 Runtime 同时加入每 turn 的小预算（1 次 Core、最多 2 次 Wikipedia、最多 3 次 read），避免模型不断换关键词耗完 loop。更关键的是，retrieved draft 不再直接返回：只要本轮做过 analogy search，Runtime 会再进行一次 no-tool quality review，专门检查是否存在 unstated motive/event、是否保留至少三条独立关系、shared mechanism 是否能脱离 surface nouns 表达，以及 analogy break 是否清楚；如果 retrieved case 很勉强，review 被明确允许直接丢弃它，改成一个 clearly labeled constructed analogy。最后一轮仍然不暴露工具，provider 如果没有返回可用文本还会再做 bounded rescue synthesis，所以正常 turn 不会重新退化成“ran out of 8 iterations”。同时把 Groq 默认模型升级为 GPT-OSS 120B，并对 decision turn 使用 high reasoning effort。新增回归测试覆盖 unmapped mechanism widening、search budget、quality-review replacement 和 rescue synthesis；完整 deterministic suite 为 49 passed、3 skipped。
