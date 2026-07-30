---
aliases:
  - "大语言模型后训练"
  - "LLM Post-Training"
  - "LLM Alignment Training"
status: source-checked
---

# LLM 后训练：SFT、RLHF 与 DPO
<!-- bilingual-en:start -->
*LLM Post-Training: SFT, RLHF, and DPO*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 把“会预测文本”的基础模型逐步变成能理解任务、遵循指令并更符合人类偏好的助手。
> **具体锚点：** 预训练模型可能自然续写问题；SFT 教它按问答格式回答，偏好优化再让“正确、相关、安全”的答案相对更可能出现。
> **核心难点：** 三阶段目标不同：预训练学分布，SFT 模仿示范，RLHF/DPO 学相对偏好；后两者不能凭空补回预训练中没有的能力或事实。
> **为什么重要：** 看到“模型更好”时，必须问清提升来自知识、指令遵循、输出风格还是评审偏好。
> **继续：** 先区分三种训练信号；若关心能力增长的来源，转到 [[Scaling laws 与计算最优训练|规模化、MoE 与分布式训练]]；若关心怎样测量，转到 [[LLM 评测]]。
> <!-- bilingual-en:start -->
> **Problem addressed:** Turn a base model that predicts text into an assistant that interprets tasks, follows instructions, and better reflects human preferences.
> **Concrete anchor:** A pretrained model may naturally continue a question; supervised fine-tuning teaches it to answer in a question–answer format, and preference optimization makes correct, relevant, and safe responses relatively more likely.
> **Central difficulty:** The stages use different signals: pretraining learns a distribution, SFT imitates demonstrations, and RLHF or DPO learns relative preferences. The later stages cannot manufacture knowledge or capabilities absent from the base model.
> **Why it matters:** When a model is said to be “better,” distinguish added knowledge from improved instruction following, output style, or agreement with evaluators.
> **Continue with:** Separate the training signals below; use [[Scaling laws 与计算最优训练|Scaling Laws and Compute-Optimal Training]] for capability growth from scale and [[LLM 评测|LLM Evaluation]] for measurement.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [[06_paper/LLM/LLM Map Index]] 及其链接的论文笔记：用于定位主题与原论文。
> - 对应 Zotero 原论文：核验架构、训练方法、实验条件与结论；论文笔记本身不替代原文。
> <!-- bilingual-en:start -->
> - [[06_paper/LLM/LLM Map Index|LLM Map Index]] and its linked paper notes locate the topic and original papers.
> - The corresponding original papers in Zotero verify architectures, training methods, experimental conditions, and conclusions; paper notes do not replace the primary papers.
> <!-- bilingual-en:end -->

## 三类训练信号先分开
<!-- bilingual-en:start -->
*First Separate the Three Training Signals*
<!-- bilingual-en:end -->

[[LLM 预训练]] 用观测到的 token 作为目标；SFT 用人工或合成的理想回答作为目标；偏好优化用同一提示下哪个回答更好作为目标。它们都能改变输出概率，却回答不同问题，不能把任一阶段笼统称为“给模型知识”。
<!-- bilingual-en:start -->
[[LLM 预训练|LLM Pretraining]] targets observed tokens, supervised fine-tuning targets human or synthetic ideal responses, and preference optimization targets which response is better for the same prompt. All three alter output probabilities, but they solve different problems and should not be described generically as “giving the model knowledge.”
<!-- bilingual-en:end -->

## SFT：把能力放进任务格式
<!-- bilingual-en:start -->
*SFT: Put Capabilities into a Task Format*
<!-- bilingual-en:end -->

监督微调（SFT）用指令—回答示范最小化目标答案的交叉熵。它能教格式、任务覆盖和语气，也可能让模型过拟合示范风格。少量高质量、覆盖明确的数据常比大量低质合成指令更有效，但结论取决于基座和评测。
<!-- bilingual-en:start -->
Supervised fine-tuning minimizes cross-entropy on instruction–response demonstrations. It can teach format, task coverage, and tone, but can also overfit demonstration style. A small, high-quality, clearly scoped dataset may outperform a large low-quality synthetic collection, although the result depends on the base model and evaluation.
<!-- bilingual-en:end -->

SFT 的每个目标 token 都像预训练一样产生交叉熵，只是数据分布变为“用户输入—理想助手输出”。若训练时只对 assistant token 计 loss，用户提示提供条件而不被模仿；若模板、角色边界或结束 token 错误，模型会学到格式噪声。
<!-- bilingual-en:start -->
Each target token in SFT produces cross-entropy as in pretraining, but the data distribution becomes “user input–ideal assistant output.” If loss is applied only to assistant tokens, the user prompt conditions the answer without being imitated. Incorrect templates, role boundaries, or end tokens teach formatting noise.
<!-- bilingual-en:end -->

## RLHF：偏好模型加策略优化
<!-- bilingual-en:start -->
*RLHF: A Preference Model plus Policy Optimization*
<!-- bilingual-en:end -->

典型流程先收集同一提示下的回答排序，训练奖励模型，再优化语言模型以提高奖励，同时用 KL 约束防止偏离参考策略过远。奖励只是人类偏好的代理；奖励黑客、标注者偏差和分布外失效都可能出现。
<!-- bilingual-en:start -->
A typical pipeline gathers rankings among responses to the same prompt, trains a reward model, and then optimizes the language model for higher reward while using a KL constraint to limit divergence from a reference policy. Reward is only a proxy for human preference; reward hacking, annotator bias, and out-of-distribution failure can all occur.
<!-- bilingual-en:end -->

奖励模型常以 Bradley–Terry 式概率拟合成对偏好：$P(y_w\succ y_l\mid x)=\sigma(r(x,y_w)-r(x,y_l))$。它学习排序差异而不是绝对真理。策略优化若把代理奖励推到训练分布之外，模型可能找到评审喜欢却并非真正有用的捷径。
<!-- bilingual-en:start -->
A reward model often fits pairwise preferences with a Bradley–Terry-style probability, $P(y_w\succ y_l\mid x)=\sigma(r(x,y_w)-r(x,y_l))$. It learns ranking differences rather than absolute truth. If policy optimization pushes the proxy reward outside its training distribution, the model may discover shortcuts that evaluators favor without being genuinely useful.
<!-- bilingual-en:end -->

## DPO 与直接偏好目标
<!-- bilingual-en:start -->
*DPO and Direct Preference Objectives*
<!-- bilingual-en:end -->

DPO 直接比较 chosen/rejected 回答相对参考模型的对数概率，不显式训练奖励模型和运行在线 RL。它简化了管线，但仍依赖偏好数据、参考策略与温度等设定；它不是“无需奖励假设”。KTO、ORPO 等改变反馈形式或损失组合，应按目标和证据比较。
<!-- bilingual-en:start -->
DPO directly compares the log probabilities of chosen and rejected responses relative to a reference model, without explicitly training a reward model or running online reinforcement learning. It simplifies the pipeline but still depends on preference data, a reference policy, and settings such as temperature; it is not free of reward assumptions. Methods such as KTO and ORPO alter feedback form or loss composition and should be compared by objective and evidence.
<!-- bilingual-en:end -->

对提示 $x$、偏好回答 $y_w$ 和非偏好回答 $y_l$，DPO 提高策略相对参考模型给予 $y_w$ 而非 $y_l$ 的概率优势。若两种回答只在长度、措辞或固定模板上系统不同，目标也会忠实学习这些伪相关，因此配对构造与长度偏差审计和公式同样重要。
<!-- bilingual-en:start -->
For prompt $x$, preferred response $y_w$, and dispreferred response $y_l$, DPO increases the policy's probability advantage for $y_w$ over $y_l$ relative to the reference model. If pairs differ systematically only in length, wording, or templates, the objective faithfully learns those spurious correlations, making pair construction and length-bias audits as important as the formula.
<!-- bilingual-en:end -->

## 对齐不是一个标量
<!-- bilingual-en:start -->
*Alignment Is Not a Single Scalar*
<!-- bilingual-en:end -->

有用、诚实、无害可能冲突，不同用户和制度也有不同偏好。拒答率提高不等于安全性提高，风格更顺滑也不等于事实更可靠。应把能力、指令遵循、事实性、安全与校准分开评估。
<!-- bilingual-en:start -->
Helpfulness, honesty, and harmlessness can conflict, and users and institutions hold different preferences. A higher refusal rate does not equal greater safety, and smoother style does not equal greater factual reliability. Capability, instruction following, factuality, safety, and calibration should be evaluated separately.
<!-- bilingual-en:end -->

“对齐”最好被写成可观察的目标向量及其边界：对哪些用户、在哪些风险类别、允许哪些帮助、错误代价是什么。一个总分会掩盖过度拒答、迎合、虚假引用和危险服从之间的取舍。
<!-- bilingual-en:start -->
Alignment is better specified as a vector of observable objectives and boundaries: for which users, in which risk categories, what assistance is allowed, and what errors cost. A single score hides trade-offs among over-refusal, sycophancy, fabricated citations, and unsafe compliance.
<!-- bilingual-en:end -->

## Worked example：偏好数据可能奖励错误特征
<!-- bilingual-en:start -->
*Worked Example: Preference Data Can Reward the Wrong Feature*
<!-- bilingual-en:end -->

假设 80% 的 chosen 回答比 rejected 回答长，但正确性只略高。模型可能先学会“更长更容易被选”，离线胜率随之上升；部署时则出现冗长、自信却无依据的答案。控制长度后重新配对，并分别报告正确性与风格，才能判断偏好学习是否抓住真正目标。
<!-- bilingual-en:start -->
Suppose 80% of chosen responses are longer than rejected responses, while correctness improves only slightly. The model may first learn that longer answers are more likely to be selected, raising offline win rate but producing verbose, confident, unsupported answers in deployment. Re-pairing after controlling length and reporting correctness separately from style reveals whether preference learning captured the intended target.
<!-- bilingual-en:end -->

一个任务若基础模型正确率接近随机，SFT 可以教会回答格式，却未必产生可靠求解能力；偏好优化甚至可能让错误答案表达得更流畅。此时需要补数据、工具、检索或能力训练，而不是只调偏好 loss。
<!-- bilingual-en:start -->
If a base model performs near chance on a task, SFT may teach the answer format without creating reliable solution ability; preference optimization may merely make incorrect answers more fluent. The remedy may require data, tools, retrieval, or capability training rather than another preference-loss adjustment.
<!-- bilingual-en:end -->

## 实际诊断
<!-- bilingual-en:start -->
*Practical Diagnosis*
<!-- bilingual-en:end -->

若模型不会做任务，先区分基础能力不足、提示/格式错位、检索信息不足和偏好训练压制。若只看单一聊天评分，容易把更长、更自信或更迎合评审的输出误认为知识增加。
<!-- bilingual-en:start -->
When a model cannot perform a task, first distinguish insufficient base capability, prompt or format mismatch, missing retrieved information, and suppression from preference training. A single chat score can mistake longer, more confident, or more evaluator-pleasing outputs for added knowledge.
<!-- bilingual-en:end -->

- SFT 后通用能力退化：比较基座与微调模型在保留集上的 loss 和能力，检查学习率、训练步数、数据窄化与灾难性遗忘。
  <!-- bilingual-en:start -->
  General capability degrades after SFT: compare base and tuned models on held-out loss and capabilities, then inspect learning rate, training steps, narrow data, and catastrophic forgetting.
  <!-- bilingual-en:end -->
- 偏好分数升而人工使用变差：审计 reward overoptimization、长度和风格偏差、评审群体以及线上提示分布。
  <!-- bilingual-en:start -->
  Preference score rises while human use worsens: audit reward overoptimization, length and style bias, evaluator population, and the online prompt distribution.
  <!-- bilingual-en:end -->
- 拒答和危险服从同时存在：按风险类别构造对照提示，分别测量正确拒答、过度拒答和错误服从，不能用总体拒答率替代。
  <!-- bilingual-en:start -->
  Refusal and unsafe compliance coexist: construct controlled prompts by risk category and separately measure correct refusal, over-refusal, and incorrect compliance rather than substituting an aggregate refusal rate.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### SFT 与偏好优化的训练信号有何不同？
<!-- bilingual-en:start -->
*How do the training signals of SFT and preference optimization differ?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> SFT 模仿给定目标答案；偏好优化比较多个回答的相对好坏并改变策略偏向。
<!-- bilingual-en:start -->
> [!answer]- Answer
> SFT imitates a provided target response; preference optimization compares the relative quality of multiple responses and changes the policy's preference.
<!-- bilingual-en:end -->

### DPO 为什么不是“完全没有奖励模型假设”？
<!-- bilingual-en:start -->
*Why is DPO not completely free of reward-model assumptions?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 它把隐式奖励与 KL 正则后的最优策略关系写进直接分类式目标，仍依赖偏好数据、参考模型和温度。
<!-- bilingual-en:start -->
> [!answer]- Answer
> It embeds the relationship between an implicit reward and a KL-regularized optimal policy in a direct classification objective, while still depending on preference data, a reference model, and temperature.
<!-- bilingual-en:end -->

### 模型更少拒答后，为什么不能直接说对齐更好？
<!-- bilingual-en:start -->
*Why can fewer refusals not directly establish better alignment?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 拒答只是一个行为维度；还需检查有用性、事实性、风险内容和错误服从，且不同目标可能互相冲突。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Refusal is only one behavioral dimension. Helpfulness, factuality, risky content, and incorrect compliance must also be examined, and these objectives can conflict.
<!-- bilingual-en:end -->

### 怎样区分“能力没学会”和“能力被后训练压制”？
<!-- bilingual-en:start -->
*How can missing capability be distinguished from capability suppressed by post-training?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 在受控提示、格式和解码下比较基座与后训练模型，加入少量示例或绕开拒答表面形式，并检查过程与答案；基座也不会时，更可能是能力不足。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Compare the base and post-trained models under controlled prompts, formats, and decoding; add demonstrations or vary refusal framing and inspect both process and answer. If the base model also fails, insufficient capability is more likely.
<!-- bilingual-en:end -->

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
- Ouyang et al. (2022) 与 Rafailov et al. (2023)：核验 RLHF 管线与 DPO 目标。
  <!-- bilingual-en:start -->
  Ouyang et al. (2022) and Rafailov et al. (2023) verify the RLHF pipeline and the DPO objective.
  <!-- bilingual-en:end -->
- [Ouyang et al. (2022), Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)：核验 demonstration SFT、偏好数据、奖励模型、PPO 与 KL 约束的公开流程。
  <!-- bilingual-en:start -->
  [Ouyang et al. (2022), Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) verifies the public pipeline of demonstration SFT, preference data, reward modeling, PPO, and KL constraints.
  <!-- bilingual-en:end -->
- [Rafailov et al. (2023), Direct Preference Optimization](https://arxiv.org/abs/2305.18290)：核验 DPO 从 KL 正则奖励最大化导出的直接偏好目标及其假设。
  <!-- bilingual-en:start -->
  [Rafailov et al. (2023), Direct Preference Optimization](https://arxiv.org/abs/2305.18290) verifies DPO's direct preference objective derived from KL-regularized reward maximization and its assumptions.
  <!-- bilingual-en:end -->
