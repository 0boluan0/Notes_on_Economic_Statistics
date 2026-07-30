---
aliases:
  - "LLM Agent"
  - "大语言模型智能体"
  - "Agentic LLM Systems"
status: source-checked
---

# LLM Agent 与工具调用
<!-- bilingual-en:start -->
*LLM Agents and Tool Use*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 让语言模型在多步任务中读取环境、选择工具、执行动作并根据结果继续，而不把一次文本生成误当成完整任务执行。
> **具体锚点：** 要比较三份最新财报，Agent 可搜索文件、提取表格、计算指标、核对单位，再把证据组织成答案。
> **核心难点：** 模型输出只是行动建议；schema、权限、真实执行结果、状态、停止条件和人工确认决定系统是否安全可靠。
> **为什么重要：** 一次错误回答只污染文本，一次错误工具调用可能修改外部世界，因此 Agent 的失败面比普通聊天更大。
> **继续：** 先用最小 observe → decide → act 循环判断是否真的需要 Agent；模型内推理与 verifier 见 [[LLM 推理与验证]]。
> <!-- bilingual-en:start -->
> **Problem addressed:** Let a language model observe an environment, select tools, execute actions, and continue from results across multiple steps instead of mistaking one text completion for task execution.
> **Concrete anchor:** To compare three current financial reports, an agent can find files, extract tables, compute metrics, verify units, and assemble an evidence-backed answer.
> **Central difficulty:** Model output is only a proposed action. Schemas, permissions, actual execution results, state, stopping rules, and human confirmation determine safety and reliability.
> **Why it matters:** A wrong answer may only corrupt text, whereas a wrong tool call can change the external world, giving agents a larger failure surface than ordinary chat.
> **Continue with:** Use the minimal observe → decide → act loop below to decide whether an agent is needed; model-internal reasoning and verification belong in [[LLM 推理与验证|LLM Reasoning and Verification]].
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - ReAct 与 Toolformer 原论文：核验 reasoning/action/observation 循环和结构化工具使用。
> - NIST AI 600-1：核验来源、访问控制、监测、人类监督和风险管理边界。
> <!-- bilingual-en:start -->
> - The ReAct and Toolformer papers verify reasoning–action–observation loops and structured tool use.
> - NIST AI 600-1 verifies boundaries for provenance, access control, monitoring, human oversight, and risk management.
> <!-- bilingual-en:end -->

## 工具调用
<!-- bilingual-en:start -->
*Tool Calling*
<!-- bilingual-en:end -->

模型生成结构化调用，宿主执行并把结果返回。可靠性来自 schema 验证、明确错误、权限边界和结果核验，不来自模型声称“我调用了工具”。高风险动作需最小权限、可回滚和必要的人类确认。
<!-- bilingual-en:start -->
The model emits a structured call, the host executes it, and the result is returned. Reliability comes from schema validation, explicit errors, permission boundaries, and result verification—not from the model claiming it used a tool. High-risk actions require least privilege, recoverability, and human confirmation where necessary.
<!-- bilingual-en:end -->

一个工具 contract 至少定义名称、用途、参数类型与约束、返回结构、错误语义、权限和幂等性。宿主必须拒绝 schema 外参数，并把真实错误送回模型；静默改写或伪造成功会破坏后续状态判断。
<!-- bilingual-en:start -->
A tool contract should at minimum define name, purpose, parameter types and constraints, return structure, error semantics, permissions, and idempotency. The host must reject out-of-schema arguments and return real errors to the model; silently rewriting a call or fabricating success corrupts subsequent state reasoning.
<!-- bilingual-en:end -->

## Agent 循环
<!-- bilingual-en:start -->
*The Agent Loop*
<!-- bilingual-en:end -->

一个最小 Agent 是 observe → decide → act → observe 的闭环。规划分解目标，记忆保存可复用状态，反思或重试处理失败。只有任务真正需要跨步骤状态或行动时才值得使用；单次函数调用不必包装成 Agent。
<!-- bilingual-en:start -->
A minimal agent is an observe → decide → act → observe loop. Planning decomposes the goal, memory preserves reusable state, and reflection or retry handles failure. An agent is worthwhile only when the task truly needs state or action across steps; one function call need not be wrapped as an agent.
<!-- bilingual-en:end -->

每轮状态应区分：用户目标、已验证事实、工具原始结果、暂定假设、已执行副作用和剩余预算。把所有历史只塞进自然语言 transcript 会让过期信息、提示注入和模型猜测混成同一可信等级。
<!-- bilingual-en:start -->
Each iteration should distinguish the user goal, verified facts, raw tool results, tentative hypotheses, executed side effects, and remaining budget. Placing all history in an undifferentiated natural-language transcript mixes stale information, prompt injection, and model guesses at one trust level.
<!-- bilingual-en:end -->

停止条件应由任务状态而非“模型觉得完成”决定：验收条件满足、预算耗尽、不可恢复错误、需要权限或用户选择。重试还要根据错误类型改变策略；原样重复相同调用只会形成循环。
<!-- bilingual-en:start -->
Stopping should depend on task state rather than the model feeling finished: acceptance criteria are met, budget is exhausted, an unrecoverable error occurs, or permission or user choice is required. A retry should change strategy according to the error type; repeating the same call unchanged merely creates a loop.
<!-- bilingual-en:end -->

## Worked example：安全地删除一个生成文件
<!-- bilingual-en:start -->
*Worked Example: Safely Delete One Generated File*
<!-- bilingual-en:end -->

用户要求删除一个临时导出。Agent 先解析并展示绝对目标路径，读取确认其类型与归属，再选择可恢复的移入废纸篓；若路径来自未受信文档、包含通配符或指向宽泛目录则停止。执行后再次检查目标不存在，并报告恢复方式。
<!-- bilingual-en:start -->
A user asks to remove one temporary export. The agent resolves and displays the absolute target path, checks its type and provenance, and chooses a recoverable move to trash. It stops if the path came from an untrusted document, contains a wildcard, or targets a broad directory. After execution it verifies absence and reports recovery options.
<!-- bilingual-en:end -->

这比“调用 delete(path)”多出的步骤不是语言模型推理，而是 action guardrail：目标解析、权限、可恢复性、执行后验证。若删除不是完成目标所必需，就不应因为模型在计划中提到清理而自动授权。
<!-- bilingual-en:start -->
The additional steps beyond calling `delete(path)` are action guardrails rather than language-model reasoning: target resolution, authorization, recoverability, and postcondition verification. If deletion is not necessary to achieve the user goal, a model mentioning cleanup in its plan does not grant permission.
<!-- bilingual-en:end -->

## 不可信输入与权限边界
<!-- bilingual-en:start -->
*Untrusted Input and Permission Boundaries*
<!-- bilingual-en:end -->

网页、邮件、检索片段和工具输出都是数据，不因出现在上下文里就成为高优先级指令。系统应标记来源、隔离可执行参数，并让权限检查发生在工具宿主层；仅写一句“忽略提示注入”不能形成边界。
<!-- bilingual-en:start -->
Web pages, email, retrieved passages, and tool output are data and do not become high-priority instructions merely by appearing in context. The system should label provenance, isolate executable parameters, and enforce permissions in the tool host; a sentence telling the model to ignore prompt injection does not create a boundary.
<!-- bilingual-en:end -->

读权限与写权限应分离，写动作再按可逆性和影响范围分级。高风险工具可以要求结构化 intent、精确资源 id、预演结果、二次确认和审计记录；敏感 secret 不应回显给模型或写入长期记忆。
<!-- bilingual-en:start -->
Read and write permissions should be separated, with write actions further classified by reversibility and impact. High-risk tools can require structured intent, exact resource identifiers, dry-run results, confirmation, and audit records. Sensitive secrets should not be exposed to the model or stored in long-term memory.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure Diagnosis*
<!-- bilingual-en:end -->

包括错误累积、循环不止、工具参数幻觉、网页/文档提示注入、把不可信输出当指令、记忆污染和在不可逆动作前未确认。限制步数和成本只能防失控，不能证明成功。
<!-- bilingual-en:start -->
Failures include error accumulation, endless loops, hallucinated tool arguments, prompt injection from pages or documents, treating untrusted output as instructions, memory contamination, and failing to confirm before irreversible actions. Step and cost limits prevent runaway behavior but do not prove task success.
<!-- bilingual-en:end -->

- 工具返回成功但任务没完成：定义并检查 postcondition，不以 HTTP 200 或调用无异常作为业务成功。
  <!-- bilingual-en:start -->
  A tool reports success but the task remains incomplete: define and check a postcondition rather than equating HTTP 200 or absence of an exception with business success.
  <!-- bilingual-en:end -->
- 重复同一失败动作：按错误分类设置重试上限与替代策略，连续相同条件应升级为阻塞。
  <!-- bilingual-en:start -->
  The same failed action repeats: classify errors, limit retries, and choose an alternative; repeated identical conditions should escalate to a blocked state.
  <!-- bilingual-en:end -->
- 记忆让后续任务持续出错：长期写入前做来源与稳定性审核，允许失效、版本化与删除。
  <!-- bilingual-en:start -->
  Memory causes persistent errors in later tasks: audit provenance and stability before long-term writes and support expiry, versioning, and deletion.
  <!-- bilingual-en:end -->
- 任务成功率高但出现一次严重越权：安全违规应作为独立硬指标，不能被平均成功率抵消。
  <!-- bilingual-en:start -->
  Task success is high but one serious unauthorized action occurs: safety violations are separate hard metrics and cannot be averaged away by success rate.
  <!-- bilingual-en:end -->

## 评测
<!-- bilingual-en:start -->
*Evaluation*
<!-- bilingual-en:end -->

评测最终任务成功率、成本、延迟、恢复能力和安全违规；再诊断规划、工具选择和执行各环节。静态问答分数不能代表在动态环境中的 Agent 能力。
<!-- bilingual-en:start -->
Evaluate final task success, cost, latency, recovery, and safety violations, then diagnose planning, tool selection, and execution stages. Static question-answering scores do not represent agent capability in a dynamic environment.
<!-- bilingual-en:end -->

可复现评测要固定环境初态、工具版本和权限，并记录完整 action/event log。结果最好分为完成、部分完成、正确阻塞、错误失败和越权，而不是只有一个“答案相似度”。
<!-- bilingual-en:start -->
Reproducible evaluation fixes the initial environment state, tool versions, and permissions and records a complete action and event log. Outcomes should distinguish completion, partial completion, correct blocking, incorrect failure, and unauthorized action rather than using one answer-similarity score.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### 什么时候不需要 Agent？
<!-- bilingual-en:start -->
*When is an agent unnecessary?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 任务能由一次模型回答或确定性函数完成、无需跨步骤环境反馈时，Agent 只增加失败面。
<!-- bilingual-en:start -->
> [!answer]- Answer
> When one model response or a deterministic function completes the task without multi-step environmental feedback, an agent only adds failure surface.
<!-- bilingual-en:end -->

### 给能删除文件的 Agent 加一步数上限是否足够安全？
<!-- bilingual-en:start -->
*Is a step limit sufficient for an agent that can delete files?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 不够；还需精确目标解析、权限限制、可恢复操作、结果检查以及高风险动作前确认。
<!-- bilingual-en:start -->
> [!answer]- Answer
> No. Exact target resolution, permission limits, recoverable operations, result checks, and confirmation before high-risk actions are also required.
<!-- bilingual-en:end -->

### 工具调用返回“成功”后为什么还要检查 postcondition？
<!-- bilingual-en:start -->
*Why must a postcondition be checked after a tool reports success?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 调用成功只说明接口接受或完成某一步；目标对象可能不对、外部状态可能未改变，或业务验收条件仍未满足。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Call success may mean only that the interface accepted or completed one step. The target may be wrong, external state may be unchanged, or business acceptance criteria may remain unmet.
<!-- bilingual-en:end -->

### 怎样判断任务应使用 Agent 还是固定 workflow？
<!-- bilingual-en:start -->
*How should one choose between an agent and a fixed workflow?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 步骤和分支若可预先穷举、错误处理明确，固定 workflow 更可控；只有需要依据新观测动态选择动作时，才增加 Agent 决策层。
<!-- bilingual-en:start -->
> [!answer]- Answer
> A fixed workflow is more controllable when steps, branches, and error handling can be enumerated in advance. Add agentic decision-making only when new observations require dynamic action choice.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- [Yao et al. (2022), ReAct](https://arxiv.org/abs/2210.03629)：核验交替 reasoning/action/observation 的任务循环与工具交互实验。
  <!-- bilingual-en:start -->
  [Yao et al. (2022), ReAct](https://arxiv.org/abs/2210.03629) verifies task loops that alternate reasoning, action, and observation and their tool-interaction experiments.
  <!-- bilingual-en:end -->
- [Schick et al. (2023), Toolformer](https://arxiv.org/abs/2302.04761)：核验语言模型学习何时、以何种参数调用外部工具的公开方法。
  <!-- bilingual-en:start -->
  [Schick et al. (2023), Toolformer](https://arxiv.org/abs/2302.04761) verifies a public method for language models to learn when and with which arguments to call external tools.
  <!-- bilingual-en:end -->
- [NIST AI 600-1, Generative Artificial Intelligence Profile](https://doi.org/10.6028/NIST.AI.600-1)：核验生成式 AI 系统的来源追踪、访问控制、监测、人类监督与风险管理要求。
  <!-- bilingual-en:start -->
  [NIST AI 600-1, Generative Artificial Intelligence Profile](https://doi.org/10.6028/NIST.AI.600-1) verifies provenance, access control, monitoring, human oversight, and risk-management requirements for generative-AI systems.
  <!-- bilingual-en:end -->
