---
aliases:
  - "LLM Evaluation"
  - "大语言模型评测"
status: source-checked
---

# LLM 评测
<!-- bilingual-en:start -->
*LLM Evaluation*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 判断模型在具体用途上是否真的更好，并定位改进来自能力、数据、提示还是评审偏差。
> **具体锚点：** 数学基准得分上升可能来自推理更强，也可能是题目出现在训练数据、采样次数更多或答案解析更宽松。
> **核心难点：** 基准是测量工具，不是能力本身；污染、饱和、评审偏好和不可复现设置会让一个分数失真。
> **为什么重要：** 没有任务定义、基线和误差分析，排行榜无法支持模型选型或安全判断。
> **继续：** 先定义实际决策和失败代价，再组合自动指标、人类/专家评审与可复现实验；RAG 和 Agent 还要分层测管线。
> <!-- bilingual-en:start -->
> **Problem addressed:** Determine whether a model is genuinely better for a particular use and locate whether improvement comes from capability, data, prompting, or evaluator bias.
> **Concrete anchor:** A higher mathematics benchmark score may reflect stronger reasoning, but it may instead come from training-set exposure, more samples, or a more permissive answer parser.
> **Central difficulty:** A benchmark is a measurement instrument, not the capability itself. Contamination, saturation, evaluator preference, and irreproducible settings can distort a score.
> **Why it matters:** Without a task definition, baseline, and error analysis, a leaderboard cannot support model selection or safety judgment.
> **Continue with:** Define the real decision and failure costs first, then combine automatic metrics, human or expert review, and reproducible experiments. Evaluate RAG and agents by pipeline stage as well as final outcome.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [[06_paper/LLM/LLM Map Index]] 及其链接的论文笔记：用于定位主题与原论文。
> - 对应 Zotero 原论文：核验架构、训练方法、实验条件与结论；论文笔记本身不替代原文。
> <!-- bilingual-en:start -->
> - [[06_paper/LLM/LLM Map Index|LLM Map Index]] and its linked paper notes locate the topic and primary papers.
> - The corresponding original papers in Zotero verify architectures, training methods, experimental conditions, and conclusions; paper notes do not replace primary papers.
> <!-- bilingual-en:end -->

## 从用途到维度
<!-- bilingual-en:start -->
*From Use Case to Evaluation Dimensions*
<!-- bilingual-en:end -->

先写清用户、任务分布、允许的工具、成本和严重失败。能力可分知识、推理、代码、指令遵循、多语；风险可分事实性、偏见、隐私、安全与稳健性。综合分数若不公开权重，会隐藏关键退步。
<!-- bilingual-en:start -->
First specify users, task distribution, allowed tools, cost, and severe failures. Capability may be separated into knowledge, reasoning, code, instruction following, and multilingual performance; risks may include factuality, bias, privacy, safety, and robustness. An aggregate score hides critical regressions when its weights are not explicit.
<!-- bilingual-en:end -->

评测设计应从决策反推：若要选客服模型，主要单位可能是一次真实对话及其解决状态，而不是独立问答；若要判断能否自动执行高风险动作，任何越权都可能是 hard failure，不能由其他题目平均抵消。
<!-- bilingual-en:start -->
Evaluation design should work backward from the decision. For customer support, the unit may be a complete real conversation and resolution state rather than an isolated question. For autonomous high-risk actions, any unauthorized action may be a hard failure that cannot be averaged away by other items.
<!-- bilingual-en:end -->

## 基准有效性与污染
<!-- bilingual-en:start -->
*Benchmark Validity and Contamination*
<!-- bilingual-en:end -->

construct validity 问“这个题集是否真的测目标能力”。训练集污染、近重复和公开答案会抬高得分；基准饱和后差异主要是噪声。使用隐藏/新鲜集、canary、去重分析和时间切分，并记录版本。
<!-- bilingual-en:start -->
Construct validity asks whether the item set actually measures the target capability. Training contamination, near duplicates, and public answers inflate scores; once a benchmark saturates, differences can be dominated by noise. Use hidden or fresh sets, canaries, deduplication analysis, and time splits, and record versions.
<!-- bilingual-en:end -->

还要区分内容污染与过程调参：即使题目不在预训练中，开发者反复查看 test error 并改 prompt、tool 或 system，就把 test 集变成了开发集。最终结论需要未参与选择的 held-out 集或嵌套评测流程。
<!-- bilingual-en:start -->
Content contamination must be separated from procedural tuning. Even if items were absent from pretraining, repeatedly inspecting test errors and changing prompts, tools, or systems turns the test set into a development set. Final claims need a held-out set untouched by selection or a nested evaluation process.
<!-- bilingual-en:end -->

## 自动指标、LLM judge 与人类评审
<!-- bilingual-en:start -->
*Automatic Metrics, LLM Judges, and Human Review*
<!-- bilingual-en:end -->

精确匹配适合有唯一规范答案，代码可用测试，生成任务常需 rubric。LLM-as-a-judge 可扩展，但受位置、长度、风格、自我偏好与提示影响；要随机顺序、校准人类标签并报告一致性。高风险领域需要合格专家。
<!-- bilingual-en:start -->
Exact match suits unique canonical answers, code can be executed against tests, and generation tasks often need a rubric. LLM-as-a-judge scales, but is affected by position, length, style, self-preference, and prompt choice. Randomize order, calibrate against human labels, and report agreement. High-risk domains require qualified experts.
<!-- bilingual-en:end -->

评分器的能力上限、偏差和方差都要测。成对比较通常比绝对 1–10 分更稳定，但仍需交换 A/B 顺序；rubric 要把事实正确、相关、完整、引用和风格拆开，避免 judge 用“更长”代替“更好”。
<!-- bilingual-en:start -->
Measure the scorer's capability ceiling, bias, and variance. Pairwise comparison is often more stable than an absolute one-to-ten score but still requires swapping A and B order. A rubric should separate factual correctness, relevance, completeness, citation, and style so the judge cannot substitute length for quality.
<!-- bilingual-en:end -->

## Worked example：2 分差异是否可信
<!-- bilingual-en:start -->
*Worked Example: Is a Two-Point Difference Credible?*
<!-- bilingual-en:end -->

模型 A 在 200 题答对 150 题，模型 B 答对 154 题。只报 75% 对 77% 会忽略二者是否在同一道题上成对变化；应保留逐题结果，用 paired bootstrap 或适当的成对检验给差异区间，并审查那几道翻转题是否来自 parser、污染或随机采样。
<!-- bilingual-en:start -->
Model A answers 150 of 200 items correctly and Model B answers 154. Reporting only 75% versus 77% ignores whether the models changed on the same items. Preserve item-level outcomes, use a paired bootstrap or suitable paired test for an interval on the difference, and inspect flipped items for parser effects, contamination, or sampling randomness.
<!-- bilingual-en:end -->

如果 B 每题采样 32 次后取多数票，而 A 只采样一次，这不是同一推理预算下的模型比较。可以分别报告固定调用预算的质量和允许各系统最佳配置的成本—质量前沿，但不能混成单一能力结论。
<!-- bilingual-en:start -->
If B samples 32 times per item and takes a majority vote while A samples once, this is not a same-compute model comparison. Report quality under a fixed inference budget and the cost–quality frontier under each system's best configuration separately rather than merging them into one capability claim.
<!-- bilingual-en:end -->

## 事实性、安全与稳健性
<!-- bilingual-en:start -->
*Factuality, Safety, and Robustness*
<!-- bilingual-en:end -->

事实评测应检查主张是否有证据及模型能否表达不确定性。安全评测既测有害服从，也测过度拒答和正常任务可用性。稳健性要改变措辞、顺序、语言和干扰信息，而不是重复同一模板。
<!-- bilingual-en:start -->
Factuality evaluation should check whether claims have evidence and whether the model expresses uncertainty appropriately. Safety evaluation measures both harmful compliance and over-refusal or ordinary-task utility. Robustness requires varying wording, order, language, and distractors rather than repeating one template.
<!-- bilingual-en:end -->

风险指标应保留严重度和分母。例如“危险服从率”要说明哪些 prompt 真正需要拒绝，“过度拒答率”要在明确良性集上测；把两类样本混成总体准确率会掩盖安全—效用 trade-off。
<!-- bilingual-en:start -->
Risk metrics should preserve severity and denominator. A harmful-compliance rate must define which prompts truly require refusal, while over-refusal should be measured on clearly benign items. Combining both into overall accuracy hides the safety–utility trade-off.
<!-- bilingual-en:end -->

## 推理、工具与 Agent 评测
<!-- bilingual-en:start -->
*Reasoning, Tool, and Agent Evaluation*
<!-- bilingual-en:end -->

优先核验最终答案或环境状态，再测步骤/成本。工具系统分解为选择工具、参数正确、执行恢复、证据使用和权限违规；多次运行报告成功率分布，不能只展示最佳轨迹。
<!-- bilingual-en:start -->
Verify the final answer or environment state first, then measure steps and cost. Decompose tool systems into tool selection, argument correctness, execution recovery, evidence use, and permission violations. Report the success distribution over repeated runs instead of showing only the best trace.
<!-- bilingual-en:end -->

RAG 还要分别报告 ingestion/index coverage、retrieval recall、ranking、context assembly、citation entailment 和有证据条件下的生成正确率；否则最终错误无法归因。对应诊断见 [[RAG（检索增强生成）]] 与 [[LLM Agent 与工具调用]]。
<!-- bilingual-en:start -->
RAG should separately report ingestion and index coverage, retrieval recall, ranking, context assembly, citation entailment, and generation accuracy conditional on available evidence; otherwise final errors cannot be attributed. See [[RAG（检索增强生成）|Retrieval-Augmented Generation]] and [[LLM Agent 与工具调用|LLM Agents and Tool Use]] for diagnostic structure.
<!-- bilingual-en:end -->

## 可复现实验与误差分析
<!-- bilingual-en:start -->
*Reproducible Experiments and Error Analysis*
<!-- bilingual-en:end -->

记录模型快照、系统提示、采样参数、工具版本、日期、数据和评分代码。给置信区间或配对差异，按失败类型抽样复核。评测集参与调参后就不再是独立测试集。
<!-- bilingual-en:start -->
Record model snapshot, system prompt, sampling parameters, tool versions, date, data, and scoring code. Provide confidence intervals or paired differences and review samples by failure type. Once an evaluation set participates in tuning, it is no longer an independent test set.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure Diagnosis*
<!-- bilingual-en:end -->

- 分数无法复现：先锁定 model snapshot、prompt、seed/采样、依赖、数据版本与 scorer，再查随机性。
  <!-- bilingual-en:start -->
  A score cannot be reproduced: first lock the model snapshot, prompt, seed and sampling, dependencies, data version, and scorer before attributing variation to randomness.
  <!-- bilingual-en:end -->
- 自动分高但人工差：抽查 parser 与 rubric，查看格式投机、长度偏差和 judge 无法识别的领域错误。
  <!-- bilingual-en:start -->
  Automatic scores are high but human quality is poor: inspect parsers and rubrics for format gaming, length bias, and domain errors beyond the judge's competence.
  <!-- bilingual-en:end -->
- 总分升但上线更差：按真实流量权重、子群、严重失败和延迟/成本重新分层。
  <!-- bilingual-en:start -->
  Aggregate score rises while deployment worsens: re-stratify by real traffic weights, subgroups, severe failures, and latency or cost.
  <!-- bilingual-en:end -->
- 多次运行方差很大：报告分布与置信区间，控制采样并分析不稳定题，而不是挑最好一次。
  <!-- bilingual-en:start -->
  Repeated runs have high variance: report distributions and confidence intervals, control sampling, and analyze unstable items rather than selecting the best run.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### 排行榜高 2 分为何未必支持选型？
<!-- bilingual-en:start -->
*Why might a two-point leaderboard advantage not support model selection?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 需知道任务是否匹配、差异不确定性、污染、推理预算、成本和关键失败；单个平均分不足。
> <!-- bilingual-en:start -->
> Task match, uncertainty in the difference, contamination, inference budget, cost, and critical failures are all required; one average score is insufficient.
> <!-- bilingual-en:end -->

### 使用 LLM judge 至少要防哪类偏差？
<!-- bilingual-en:start -->
*Which biases must at least be controlled when using an LLM judge?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 位置、长度/风格、自我偏好和提示敏感；应随机化、用 rubric 并与人类校准。
> <!-- bilingual-en:start -->
> Control position, length and style, self-preference, and prompt sensitivity through randomization, a rubric, and calibration against humans.
> <!-- bilingual-en:end -->

### Agent 轨迹写得很漂亮但最终文件错误，应如何计分？
<!-- bilingual-en:start -->
*How should a beautiful agent trace be scored when the final file is wrong?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 任务失败；先以可验证的最终环境状态计分，再用轨迹定位工具选择或执行错误。
> <!-- bilingual-en:start -->
> The task failed. Score the verifiable final environment state first, then use the trace to locate tool-selection or execution errors.
> <!-- bilingual-en:end -->

### 一个 test set 被用来改了十次 prompt 后，为什么不再是独立测试？
<!-- bilingual-en:start -->
*Why is a test set no longer independent after being used to revise a prompt ten times?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 选择过程已经根据 test error 适配 prompt，导致对该集合过拟合；需要未参与选择的新 held-out 数据评价最终方案。
> <!-- bilingual-en:start -->
> The selection process adapted the prompt to test errors and overfit that set. New held-out data untouched by selection are needed for final evaluation.
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
- [Liang et al. (2022), Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)：核验按 scenario、metric 与模型多维透明评测及稳健性、校准、公平、效率框架。
  <!-- bilingual-en:start -->
  [Liang et al. (2022), Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) verifies transparent multidimensional evaluation across scenarios, metrics, and models, including robustness, calibration, fairness, and efficiency.
  <!-- bilingual-en:end -->
- [Zheng et al. (2023), Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)：核验 LLM judge 的可扩展性以及位置、冗长和自我增强偏差。
  <!-- bilingual-en:start -->
  [Zheng et al. (2023), Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) verifies the scalability of LLM judges and their position, verbosity, and self-enhancement biases.
  <!-- bilingual-en:end -->
- [Srivastava et al. (2022), Beyond the Imitation Game Benchmark](https://arxiv.org/abs/2206.04615)：核验大规模多任务 benchmark、任务异质性与能力测量边界。
  <!-- bilingual-en:start -->
  [Srivastava et al. (2022), Beyond the Imitation Game Benchmark](https://arxiv.org/abs/2206.04615) verifies large-scale multitask benchmarking, task heterogeneity, and boundaries of capability measurement.
  <!-- bilingual-en:end -->
