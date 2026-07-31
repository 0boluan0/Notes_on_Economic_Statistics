---
aliases:
  - "推理模型"
  - "Reasoning Models"
  - "LLM Reasoning and Verification"
status: source-checked
---

# LLM 推理与验证
<!-- bilingual-en:start -->
*LLM Reasoning and Verification*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 让模型在复杂问题上分解、搜索候选并验证结果，而不是只凭一次续写直接给答案。
> **具体锚点：** 解一道多步数学题时，模型可生成若干候选步骤，用独立计算或规则检查结果，再根据失败反馈重试。
> **核心难点：** 更长的思考不保证正确，表面流畅的 reasoning trace 也不是证明；验证器必须检查真正决定答案的条件，并与生成器保持足够独立。
> **为什么重要：** 它把“答案像不像推理”改成“搜索和验证是否提高最终正确率”，避免把可读轨迹误当可靠性。
> **继续：** 先区分模型内推理与环境循环，再用 [[LLM 评测#推理、工具与 Agent 评测|任务级结果]] 检验，而不是只看轨迹是否像人。
> <!-- bilingual-en:start -->
> **Problem addressed:** Improve performance on complex tasks by decomposing, searching, and verifying instead of producing an answer in one unexamined continuation.
> **Concrete anchor:** For a multi-step mathematics problem, a model may generate candidate steps, verify a numerical result, and retry after failure. An agent extends such loops to search, code, and environmental actions.
> **Central difficulty:** Longer reasoning does not guarantee correctness, and a textual plan is not reliable state. Verification quality and compute allocation determine reasoning reliability; tool permissions and environmental control belong to agent engineering.
> **Why it matters:** Model reasoning and agent systems are often conflated. The former improves problem-solving strategy, while the latter additionally manages real-world action and risk.
> **Continue with:** Separate internal reasoning from [[LLM Agent 与工具调用|LLM Agents and Tool Use]], then evaluate task outcomes through [[LLM 评测#推理、工具与 Agent 评测|LLM Evaluation]] rather than judging whether a trace merely looks human.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [[06_paper/LLM/LLM Map Index]] 及其链接的论文笔记：用于定位主题与原论文。
> - 对应 Zotero 原论文：核验架构、训练方法、实验条件与结论；论文笔记本身不替代原文。
> <!-- bilingual-en:start -->
> - [[06_paper/LLM/LLM Map Index|LLM Map Index]] and its linked paper notes locate the topic and primary papers.
> - The corresponding original papers in Zotero verify architectures, training methods, experimental conditions, and conclusions; paper notes do not replace primary papers.
> <!-- bilingual-en:end -->

## 先区分答案、轨迹与验证
<!-- bilingual-en:start -->
*First Separate Answers, Traces, and Verification*
<!-- bilingual-en:end -->

最终答案是任务输出，reasoning trace 是模型生成的中间 token，verifier 是独立或半独立地判断候选是否满足约束的过程。轨迹可以帮助搜索，却不是答案正确性的证书；验证器只有在错误相关性足够低、检查条件足够强时才增加可靠性。
<!-- bilingual-en:start -->
The final answer is the task output, a reasoning trace is an intermediate token sequence, and a verifier is a process that judges whether a candidate satisfies constraints independently or semi-independently. A trace can support search but is not a certificate of correctness; a verifier improves reliability only when its errors are sufficiently uncorrelated and its checks sufficiently strong.
<!-- bilingual-en:end -->

## Chain-of-thought 与采样
<!-- bilingual-en:start -->
*Chain of Thought and Sampling*
<!-- bilingual-en:end -->

chain-of-thought 提示诱发中间推理文本；self-consistency 对多条路径投票。它们可提高部分任务表现，但中间文本不一定忠实反映模型真正使用的机制，也可能流畅地放大早期错误。
<!-- bilingual-en:start -->
Chain-of-thought prompting elicits intermediate reasoning text, while self-consistency votes across multiple paths. These techniques improve some tasks, but the intermediate text need not faithfully reflect the mechanism producing the answer and can fluently amplify an early error.
<!-- bilingual-en:end -->

self-consistency 只有在错误路径不完全相关时才有效。若所有样本都受同一个错误假设、污染示例或系统性算术偏差影响，多数票会把共同错误放大。采样数量还应计入 test-time compute 与延迟。
<!-- bilingual-en:start -->
Self-consistency works only when erroneous paths are not perfectly correlated. If all samples share a false assumption, contaminated example, or systematic arithmetic bias, majority voting amplifies the common error. The number of samples must also be counted as test-time compute and latency.
<!-- bilingual-en:end -->

## 验证器与 reasoning RL
<!-- bilingual-en:start -->
*Verifiers and Reasoning Reinforcement Learning*
<!-- bilingual-en:end -->

结果奖励只判断最终答案，过程奖励对中间步骤给信号；外部 verifier 可筛选候选。reasoning RL 让模型在可验证任务上学习探索和分配更多计算，但可能过拟合奖励、基准格式或验证器漏洞。
<!-- bilingual-en:start -->
Outcome rewards judge only final answers, process rewards supervise intermediate steps, and an external verifier can filter candidates. Reasoning reinforcement learning can teach exploration and additional compute allocation on verifiable tasks, but may overfit the reward, benchmark format, or verifier loopholes.
<!-- bilingual-en:end -->

可执行代码、形式证明检查器或具有唯一答案的数学题提供相对强的自动验证；开放论证和事实问答往往只有不完整代理。验证强度应从“格式可解析”到“局部约束成立”再到“最终语义正确”分层，不要把通过单元测试等同于完全正确。
<!-- bilingual-en:start -->
Executable code, formal proof checkers, and mathematics problems with unique answers provide relatively strong automatic verification; open-ended arguments and factual questions usually have incomplete proxies. Verification strength should be layered from parseable format to local constraints to final semantic correctness, rather than equating passing unit tests with complete correctness.
<!-- bilingual-en:end -->

## Test-time compute 怎样发挥作用
<!-- bilingual-en:start -->
*How Test-Time Compute Helps*
<!-- bilingual-en:end -->

额外推理计算可用于更长单轨迹、并行候选、树搜索、反思重写或 verifier 排序。最佳分配取决于任务：一道可独立核验的题常适合多候选后筛选；需要连贯长计划的任务可能更依赖保留状态和阶段检查。
<!-- bilingual-en:start -->
Additional inference compute can fund a longer single trace, parallel candidates, tree search, reflective rewriting, or verifier ranking. The best allocation depends on the task: independently checkable problems often benefit from candidate generation followed by selection, while long coherent plans may depend more on persistent state and stage-level checks.
<!-- bilingual-en:end -->

报告“模型思考更久”时至少记录生成 token、候选数、verifier 调用、总 FLOPs/成本和准确率。若只增加 token 而错误率不降，可能是搜索缺乏多样性、验证器过弱或模型根本缺少所需知识。
<!-- bilingual-en:start -->
When reporting that a model “thinks longer,” record generated tokens, candidate count, verifier calls, total FLOPs or cost, and accuracy. If more tokens do not reduce errors, search may lack diversity, the verifier may be weak, or the model may lack required knowledge.
<!-- bilingual-en:end -->

## Worked example：多数票何时失败
<!-- bilingual-en:start -->
*Worked Example: When Majority Voting Fails*
<!-- bilingual-en:end -->

假设独立路径各有 70% 概率正确，取五条路径多数票的正确率约为 $\sum_{k=3}^{5}\binom{5}{k}0.7^k0.3^{5-k}\approx83.7\%$。但若错误高度相关，例如所有路径都误读同一条件，则独立假设失效，投票可能仍接近 70% 甚至更差。
<!-- bilingual-en:start -->
Suppose independent paths are each correct with probability 70%. A five-path majority vote succeeds with probability $\sum_{k=3}^{5}\binom{5}{k}0.7^k0.3^{5-k}\approx83.7\%$. If errors are highly correlated—for example, every path misreads the same condition—the independence assumption fails and voting may remain near 70% or become worse.
<!-- bilingual-en:end -->

若有一个真正检查约束的 verifier，候选质量和 verifier accuracy 共同决定收益。只让同一模型阅读自己的答案并回答“是否正确”，常产生相关偏差；执行、反例搜索或外部证据通常提供更独立的信号。
<!-- bilingual-en:start -->
With a verifier that genuinely checks constraints, gains jointly depend on candidate quality and verifier accuracy. Asking the same model merely to reread its answer and declare whether it is correct often preserves correlated bias; execution, counterexample search, or external evidence usually provides a more independent signal.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure Diagnosis*
<!-- bilingual-en:end -->

- reasoning token 增加但正确率不变：检查候选多样性、错误相关性、停止规则和 verifier 区分力。
  <!-- bilingual-en:start -->
  Reasoning tokens increase without accuracy gains: inspect candidate diversity, error correlation, stopping rules, and verifier discrimination.
  <!-- bilingual-en:end -->
- 轨迹合理但最终答案错：逐步执行可检验约束，定位第一处错误，不要以叙事连贯替代验证。
  <!-- bilingual-en:start -->
  The trace sounds plausible but the answer is wrong: execute checkable constraints step by step and locate the first error instead of substituting narrative coherence for verification.
  <!-- bilingual-en:end -->
- benchmark 提升只在固定格式出现：改写题面、改变答案顺序、加入分布外实例并检查训练污染。
  <!-- bilingual-en:start -->
  Benchmark gains appear only in one fixed format: paraphrase prompts, vary answer order, add out-of-distribution instances, and inspect training contamination.
  <!-- bilingual-en:end -->
- verifier score 高但真实质量低：构造对抗错误与隐藏测试，检查 reward hacking 和验证覆盖范围。
  <!-- bilingual-en:start -->
  Verifier scores are high but real quality is low: construct adversarial errors and hidden tests to inspect reward hacking and verification coverage.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### 为什么一条看似合理的 chain-of-thought 不是正确性的证明？
<!-- bilingual-en:start -->
*Why is a plausible chain of thought not proof of correctness?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 它可能是事后合理化或在中间步骤悄悄出错；仍需可独立核验的结果、约束或 verifier。
> <!-- bilingual-en:start -->
> It may be a post-hoc rationalization or contain a hidden intermediate error; independently checkable results, constraints, or a verifier are still required.
> <!-- bilingual-en:end -->

### self-consistency 在什么条件下最可能有用？
<!-- bilingual-en:start -->
*Under which conditions is self-consistency most likely to help?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 单条路径有一定正确率、采样能产生实质不同路径、错误不高度相关，而且最终答案可可靠聚合。
> <!-- bilingual-en:start -->
> Individual paths must have some competence, sampling must produce substantively different paths, errors must not be highly correlated, and final answers must be aggregable reliably.
> <!-- bilingual-en:end -->

### 为什么可执行 verifier 通常比“再问模型一次是否正确”更强？
<!-- bilingual-en:start -->
*Why is an executable verifier usually stronger than asking the model again whether it is correct?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 执行可对明确约束给独立结果；同一模型的自评容易重复生成时的偏差，但执行也只覆盖被编码的测试与条件。
> <!-- bilingual-en:start -->
> Execution gives an independent result for explicit constraints. Self-evaluation by the same model can repeat generation bias, although execution still covers only encoded tests and conditions.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- [[06_paper/LLM/LLM Map Index]] 及其链接的论文笔记：用于定位主题与原论文。
  <!-- bilingual-en:start -->
  [[06_paper/LLM/LLM Map Index|LLM Map Index]] and its linked paper notes locate the topic and primary papers.
  <!-- bilingual-en:end -->
- 对应 Zotero 原论文：核验架构、训练方法、实验条件与结论；论文笔记本身不替代原文。
  <!-- bilingual-en:start -->
  The corresponding original papers in Zotero verify architectures, training methods, experimental conditions, and conclusions; paper notes do not replace primary papers.
  <!-- bilingual-en:end -->
- [Wei et al. (2022), Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)：核验中间推理提示在算术、常识与符号任务上的实验效果与设置。
  <!-- bilingual-en:start -->
  [Wei et al. (2022), Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) verifies experimental effects and settings of intermediate-reasoning prompts on arithmetic, commonsense, and symbolic tasks.
  <!-- bilingual-en:end -->
- [Wang et al. (2022), Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)：核验多路径采样与答案聚合的 self-consistency 方法。
  <!-- bilingual-en:start -->
  [Wang et al. (2022), Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) verifies self-consistency through multi-path sampling and answer aggregation.
  <!-- bilingual-en:end -->
- [Lightman et al. (2023), Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)：核验 outcome 与 process supervision、过程奖励模型及 best-of-N 选择。
  <!-- bilingual-en:start -->
  [Lightman et al. (2023), Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) verifies outcome and process supervision, process reward models, and best-of-N selection.
  <!-- bilingual-en:end -->
- [DeepSeek-AI et al. (2025), DeepSeek-R1](https://arxiv.org/abs/2501.12948)：核验可验证任务上 reasoning-oriented reinforcement learning、采样计算与行为边界的公开实例。
  <!-- bilingual-en:start -->
  [DeepSeek-AI et al. (2025), DeepSeek-R1](https://arxiv.org/abs/2501.12948) verifies a public instance of reasoning-oriented reinforcement learning, sampling compute, and behavioral boundaries on verifiable tasks.
  <!-- bilingual-en:end -->
