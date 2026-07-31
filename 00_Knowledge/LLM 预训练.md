---
aliases:
  - "大语言模型预训练"
  - "Language Model Pretraining"
  - "LLM Pretraining"
status: source-checked
---

# LLM 预训练
<!-- bilingual-en:start -->
*LLM Pretraining*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 在没有逐条任务标注的情况下，让模型从大规模 token 序列中学习语言规律、世界知识和可迁移的表示与能力。
> **具体锚点：** 给定“巴黎是法国的”，模型反复学习提高下一个 token“首都”的概率；海量不同上下文共同塑造其参数。
> **核心难点：** 训练目标很简单，但结果由数据组成、去重与污染、tokenizer、模型规模、训练 token、优化稳定性和算力分配共同决定。
> **为什么重要：** 后训练主要改变能力如何被调用和偏好哪种输出；基础知识与大部分通用能力首先受预训练决定。
> **继续：** 先理解 next-token 交叉熵和数据管线，再读 [[Scaling laws 与计算最优训练]]；助手行为见 [[LLM 后训练：SFT、RLHF 与 DPO]]。
> <!-- bilingual-en:start -->
> **Problem addressed:** Learn language regularities, world knowledge, and transferable representations and capabilities from large token sequences without task-by-task labels.
> **Concrete anchor:** Given the prefix “Paris is France's,” the model repeatedly learns to raise the probability of the next token “capital”; parameters are shaped by vast numbers of such contexts.
> **Central difficulty:** The objective is simple, but outcomes jointly depend on data composition, deduplication and contamination, the tokenizer, model size, training tokens, optimization stability, and compute allocation.
> **Why it matters:** Post-training mainly changes how capabilities are elicited and which outputs are preferred; foundational knowledge and much general capability are first determined by pretraining.
> **Continue with:** Understand next-token cross-entropy and the data pipeline below, then use [[Scaling laws 与计算最优训练|Scaling Laws and Compute-Optimal Training]] for budget allocation and [[LLM 后训练：SFT、RLHF 与 DPO|LLM Post-Training: SFT, RLHF, and DPO]] for assistant behavior.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - Transformer 的自回归目标与公开大模型训练论文：核验目标函数、训练设置与规模变量。
> - Chinchilla 论文：核验固定计算预算下参数量与训练 token 的联合分配。
> <!-- bilingual-en:start -->
> - The Transformer autoregressive objective and public large-model training papers verify the objective, training setup, and scale variables.
> - The Chinchilla paper verifies joint allocation between parameters and training tokens under a fixed compute budget.
> <!-- bilingual-en:end -->

## 自回归目标怎样产生训练信号
<!-- bilingual-en:start -->
*How the Autoregressive Objective Produces a Training Signal*
<!-- bilingual-en:end -->

大规模文本上的自监督 next-token 或去噪目标无需人工逐条标注。模型学习统计结构、表示和可迁移能力。数据组成、去重、质量过滤、token 数与计算预算都会改变结果；“参数更多”不是唯一变量。
<!-- bilingual-en:start -->
Self-supervised next-token or denoising objectives on large text collections do not require manual labels for each example. The model learns statistical structure, representations, and transferable capabilities. Data composition, deduplication, quality filtering, token count, and compute budget all change the result; parameter count is not the only variable.
<!-- bilingual-en:end -->

对 decoder-only 模型，训练样本 $x_1,\ldots,x_T$ 的平均负对数似然为：
<!-- bilingual-en:start -->
For a decoder-only model, the mean negative log-likelihood of a training sequence $x_1,\ldots,x_T$ is:
<!-- bilingual-en:end -->

$$L(\theta)=-\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).$$

teacher forcing 让模型在位置 $t$ 读取真实前缀 $x_{<t}$，因果掩码防止读取目标之后的 token。反向传播不是把句子逐字存入一个表，而是调整共享参数，使许多上下文中的预测误差同时下降。
<!-- bilingual-en:start -->
Teacher forcing lets the model read the true prefix $x_{<t}$ at position $t$, while the causal mask prevents access to later tokens. Backpropagation does not store each sentence in a lookup table; it adjusts shared parameters so prediction errors fall across many contexts at once.
<!-- bilingual-en:end -->

perplexity 是平均交叉熵的指数 $\exp(L)$，可理解为模型在当前 tokenization 与数据分布下的平均有效选择数。它适合同一数据、tokenizer 和预处理条件下比较语言建模，却不能直接等同于事实性、推理能力或用户满意度。
<!-- bilingual-en:start -->
Perplexity is the exponential of mean cross-entropy, $\exp(L)$, and can be interpreted as an average effective number of choices under the current tokenization and data distribution. It compares language modeling under matched data, tokenizer, and preprocessing conditions, but it is not equivalent to factuality, reasoning ability, or user satisfaction.
<!-- bilingual-en:end -->

## 数据管线是训练目标的一部分
<!-- bilingual-en:start -->
*The Data Pipeline Is Part of the Training Objective*
<!-- bilingual-en:end -->

网页、书籍、代码和专业语料通常要经过许可与来源判断、语言识别、内容抽取、质量过滤、去重、安全过滤、混合权重和 tokenization。模型实际优化的是这个处理后分布，不是抽象的“互联网”。
<!-- bilingual-en:start -->
Web pages, books, code, and domain corpora generally pass through licensing and provenance decisions, language identification, content extraction, quality filtering, deduplication, safety filtering, mixture weighting, and tokenization. The model optimizes the resulting processed distribution, not an abstract “internet.”
<!-- bilingual-en:end -->

去重既减少重复样本支配梯度，也降低训练—测试污染和逐字记忆风险；但近重复识别的阈值过强会删除有价值的独立表达，过弱则保留模板、镜像和 benchmark 答案。数据审计应同时报告来源比例、过滤规则和已知污染，而不只报告 token 总数。
<!-- bilingual-en:start -->
Deduplication reduces repeated examples dominating gradients and lowers training–test contamination and verbatim memorization risk. An overly aggressive near-duplicate threshold can remove valuable independent expressions, while a weak one retains templates, mirrors, and benchmark answers. A data audit should report source proportions, filtering rules, and known contamination alongside total token count.
<!-- bilingual-en:end -->

## 参数、token 与训练计算
<!-- bilingual-en:start -->
*Parameters, Tokens, and Training Compute*
<!-- bilingual-en:end -->

模型参数决定可用表示容量，训练 token 决定模型获得多少更新证据，计算预算限制二者的组合。固定训练 FLOPs 时，一味增大参数会减少可见 token；固定模型时重复过多低价值数据又会降低边际收益。联合分配见 [[Scaling laws 与计算最优训练]]。
<!-- bilingual-en:start -->
Model parameters determine representational capacity, training tokens determine the evidence available for updates, and compute constrains their combination. Under fixed training FLOPs, enlarging the model reduces the tokens it can see; with a fixed model, excessive repetition of low-value data yields diminishing returns. See [[Scaling laws 与计算最优训练|Scaling Laws and Compute-Optimal Training]] for joint allocation.
<!-- bilingual-en:end -->

训练还受 batch size、学习率、warmup、优化器状态、数值精度与梯度裁剪影响。规模扩大后，短暂 loss spike 可能放大并毁掉昂贵训练，因此检查点、数据批次追踪和可恢复性属于训练设计，而不是部署之后才考虑的运维细节。
<!-- bilingual-en:start -->
Training also depends on batch size, learning rate, warmup, optimizer state, numerical precision, and gradient clipping. At scale, a short loss spike can amplify and ruin an expensive run, so checkpoints, batch provenance, and recoverability are part of training design rather than post-deployment operations.
<!-- bilingual-en:end -->

## Worked example：从四个候选 token 看 loss
<!-- bilingual-en:start -->
*Worked Example: Loss over Four Candidate Tokens*
<!-- bilingual-en:end -->

若正确 token 的预测概率从 $0.10$ 提高到 $0.50$，该位置的 loss 从 $-\log 0.10\approx2.303$ 降到 $-\log0.50\approx0.693$。训练并不直接奖励“答案听起来好”，只奖励真实下一个 token 的概率增加；能力是这一局部目标在巨大、多样语料和模型容量上累积形成的结果。
<!-- bilingual-en:start -->
If the predicted probability of the correct token rises from $0.10$ to $0.50$, the position loss falls from $-\log0.10\approx2.303$ to $-\log0.50\approx0.693$. Training does not directly reward an answer for sounding good; it rewards increased probability of the observed next token. Capabilities emerge as this local objective accumulates across vast, diverse data and model capacity.
<!-- bilingual-en:end -->

设两个数据源 A、B 各有一百万 token，但采样权重为 80% 与 20%。训练一千万 token 时，期望约八百万次来自 A、两百万次来自 B；“原始数据量相同”不意味着对参数影响相同。混合权重是隐含课程设计。
<!-- bilingual-en:start -->
Suppose sources A and B each contain one million tokens but receive sampling weights of 80% and 20%. Across ten million training tokens, approximately eight million are expected from A and two million from B. Equal raw dataset sizes therefore do not imply equal influence on parameters; mixture weights define an implicit curriculum.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure Diagnosis*
<!-- bilingual-en:end -->

- 训练 loss 降而验证 loss 升：检查重复 epoch、域偏移、近重复泄漏和过拟合，不要只延长训练。
  <!-- bilingual-en:start -->
  Training loss falls while validation loss rises: inspect repeated epochs, domain shift, near-duplicate leakage, and overfitting instead of simply training longer.
  <!-- bilingual-en:end -->
- benchmark 突然异常高：搜索训练语料中的题目、答案和模板近重复，并比较时间切分或污染控制后的评测。
  <!-- bilingual-en:start -->
  A benchmark score is unexpectedly high: search the training corpus for questions, answers, and template-level near duplicates, then compare against time-split or contamination-controlled evaluations.
  <!-- bilingual-en:end -->
- 跨语言能力薄弱：检查语言 token 占比、tokenizer 碎片化和高质量语料，而不是仅靠后续翻译式 SFT 补救。
  <!-- bilingual-en:start -->
  Cross-language capability is weak: inspect language-token proportions, tokenizer fragmentation, and high-quality corpora instead of relying only on translation-style supervised fine-tuning later.
  <!-- bilingual-en:end -->
- 比较两个预训练方案只看参数量：补齐训练 token、数据组成、有效/总计算、模型架构、精度和评测协议。
  <!-- bilingual-en:start -->
  Two pretraining runs are compared only by parameter count: add training tokens, data composition, effective and total compute, architecture, precision, and evaluation protocol.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### 用自己的话解释：为什么 next-token prediction 能学到比续写格式更多的能力？
<!-- bilingual-en:start -->
*Explain in your own words why next-token prediction can learn more than continuation formatting.*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 要在多样文本中持续预测正确 token，模型必须压缩语法、语义、事实、文体和跨段依赖等可复用规律；这些内部表示可迁移到很多任务，但目标本身不保证其可靠调用。
> <!-- bilingual-en:start -->
> Consistently predicting tokens across diverse text requires compressing reusable regularities in syntax, semantics, facts, style, and long-range dependence. Those representations transfer to many tasks, although the objective does not guarantee reliable elicitation.
> <!-- bilingual-en:end -->

### perplexity 更低为什么不能直接证明助手更好？
<!-- bilingual-en:start -->
*Why does lower perplexity not directly prove that an assistant is better?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> perplexity 测量匹配数据分布下的 token 预测；助手质量还包含指令遵循、事实性、安全、校准和交互效用，而且不同 tokenizer 下数值不可直接比。
> <!-- bilingual-en:start -->
> Perplexity measures token prediction on a matched data distribution. Assistant quality also involves instruction following, factuality, safety, calibration, and interactive utility, and values under different tokenizers are not directly comparable.
> <!-- bilingual-en:end -->

### 固定计算预算下把参数翻倍时，还应追问什么？
<!-- bilingual-en:start -->
*When parameter count doubles under a fixed compute budget, what else must be asked?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 每 token 计算增加后还能训练多少 token、数据是否足够多样、是否进入训练不足区，以及生命周期推理成本是否改变最优选择。
> <!-- bilingual-en:start -->
> Ask how many tokens remain affordable after per-token compute rises, whether data remain sufficiently diverse, whether the model becomes undertrained, and whether lifecycle inference cost changes the optimum.
> <!-- bilingual-en:end -->

### 数据去重过强和过弱分别会造成什么问题？
<!-- bilingual-en:start -->
*What problems arise from overly strong and overly weak deduplication?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 过强会删除独立但相似的有价值表达并缩窄覆盖；过弱会让模板与镜像重复支配训练，增加记忆和评测污染。
> <!-- bilingual-en:start -->
> Excessive deduplication removes valuable independent but similar expressions and narrows coverage. Weak deduplication lets templates and mirrors dominate training and increases memorization and evaluation contamination.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- [Vaswani et al. (2017), Attention Is All You Need](https://arxiv.org/abs/1706.03762)：核验自回归序列建模、teacher forcing 与并行训练的架构基础。
  <!-- bilingual-en:start -->
  [Vaswani et al. (2017), Attention Is All You Need](https://arxiv.org/abs/1706.03762) verifies the architectural basis for autoregressive sequence modeling, teacher forcing, and parallel training.
  <!-- bilingual-en:end -->
- [Brown et al. (2020), Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)：核验大规模自回归预训练、数据混合、训练设置及能力评测的公开实例。
  <!-- bilingual-en:start -->
  [Brown et al. (2020), Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) verifies a public instance of large-scale autoregressive pretraining, data mixtures, training settings, and capability evaluation.
  <!-- bilingual-en:end -->
- [Hoffmann et al. (2022), Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)：核验模型参数、训练 token 与固定计算预算的联合关系。
  <!-- bilingual-en:start -->
  [Hoffmann et al. (2022), Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) verifies the joint relationship among model parameters, training tokens, and fixed compute budgets.
  <!-- bilingual-en:end -->
