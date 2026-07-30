---
aliases:
  - "Scaling Laws for Neural Language Models"
  - "LLM Scaling Laws"
  - "大模型缩放定律"
status: source-checked
---

# Scaling laws 与计算最优训练
<!-- bilingual-en:start -->
*Scaling Laws and Compute-Optimal Training*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 决定有限算力下该训练多大的模型、喂多少数据，并把一次超大训练拆到许多设备上。
> **具体锚点：** 同样的计算预算若全用来扩大参数却不给足 token，模型可能“训练不足”；Chinchilla 一类结果正是修正这种分配。
> **核心难点：** scaling law 是特定数据与训练区间内的经验关系，不是自然定律；MoE 降低每 token 激活计算，却增加路由、通信和容量管理。
> **为什么重要：** 参数量、总训练 FLOPs、推理成本和实际质量不是同一个排行。
> **继续：** 先用预算约束理解 dense scaling，再看 MoE 和并行；部署端转到 [[LLM 推理效率]]。
> <!-- bilingual-en:start -->
> **Problem addressed:** Decide how large a model to train and how much data to use under limited compute; executing that plan across devices is handled separately by distributed training.
> **Concrete anchor:** Spending the same compute budget only on more parameters while withholding tokens can leave a model undertrained; Chinchilla-style results correct that allocation.
> **Central difficulty:** A scaling law is an empirical relationship within a particular data and training regime, not a law of nature. Sparse experts and distributed execution change cost accounting without automatically changing the underlying evidence.
> **Why it matters:** Parameter count, total training FLOPs, inference cost, and actual quality do not define the same ranking.
> **Continue with:** Use the budget constraint below for dense scaling, then see [[Mixture of Experts（MoE）|Mixture of Experts]] and [[大模型分布式训练|Distributed Training for Large Models]]; deployment belongs in [[LLM 推理效率|LLM Inference Efficiency]].
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [[06_paper/LLM/LLM Map Index]] 及其链接的论文笔记：用于定位主题与原论文。
> - 对应 Zotero 原论文：核验架构、训练方法、实验条件与结论；论文笔记本身不替代原文。
> <!-- bilingual-en:start -->
> - [[06_paper/LLM/LLM Map Index|LLM Map Index]] and its linked paper notes locate the topic and primary papers.
> - The corresponding original papers in Zotero verify architectures, training methods, experimental conditions, and conclusions; paper notes do not replace primary papers.
> <!-- bilingual-en:end -->

## Scaling laws
<!-- bilingual-en:start -->
*Scaling Laws*
<!-- bilingual-en:end -->

经验上，测试损失常随模型规模、数据量和计算量呈近似幂律下降。它帮助外推和分配实验，但只在观测区间、数据分布与训练配方近似不变时可信；能力指标还可能出现阈值、污染或测量噪声。
<!-- bilingual-en:start -->
Empirically, test loss often falls approximately as a power law with model size, dataset size, and compute. This helps extrapolation and experiment allocation, but is credible only within observed regimes where data distribution and training recipe remain broadly comparable. Capability metrics can additionally show thresholds, contamination, or measurement noise.
<!-- bilingual-en:end -->

一种简化表达是把可约损失写为 $L(N,D)\approx L_\infty+aN^{-\alpha}+bD^{-\beta}$：$N$ 表示参数量，$D$ 表示训练 token，$L_\infty$ 是当前数据分布的不可约项。指数和常数必须从相近配方的实验拟合，不能从另一模型族直接搬用。
<!-- bilingual-en:start -->
A simplified expression writes reducible loss as $L(N,D)\approx L_\infty+aN^{-\alpha}+bD^{-\beta}$, where $N$ is parameter count, $D$ is training tokens, and $L_\infty$ is an irreducible term for the current distribution. Exponents and constants must be fitted on experiments with comparable recipes, not copied from another model family.
<!-- bilingual-en:end -->

## Compute-optimal training
<!-- bilingual-en:start -->
*Compute-Optimal Training*
<!-- bilingual-en:end -->

固定训练 FLOPs 时，要在参数 $N$ 与训练 token $D$ 之间分配。早期模型常偏大而数据不足；compute-optimal 研究表明更小模型配更多 token 可能在相同训练预算下达到更低损失。若模型会被大量调用，训练最优还需加入推理生命周期成本。
<!-- bilingual-en:start -->
With fixed training FLOPs, compute must be allocated between parameters $N$ and training tokens $D$. Early models were often too large and data-starved; compute-optimal studies show that a smaller model trained on more tokens can achieve lower loss under the same training budget. If the model will be queried heavily, the optimum must also include lifecycle inference cost.
<!-- bilingual-en:end -->

对常见 dense Transformer，粗略训练计算与 $ND$ 成正比，因此预算约束可写成 $C\approx kND$。compute-optimal 不是把 $N$ 和 $D$ 机械设为相等，而是在该约束下最小化经实验拟合的损失；架构、数据质量、重复 epoch 和训练稳定性改变拟合结果。
<!-- bilingual-en:start -->
For a typical dense Transformer, rough training compute is proportional to $ND$, yielding a budget constraint $C\approx kND$. Compute optimality does not mechanically set $N$ equal to $D$; it minimizes an empirically fitted loss under the constraint. Architecture, data quality, repeated epochs, and optimization stability alter that fit.
<!-- bilingual-en:end -->

## 数据、模型与算力
<!-- bilingual-en:start -->
*Data, Model, and Compute*
<!-- bilingual-en:end -->

有效 token 不是原始字节数：去重、质量、语言/领域混合、污染和重复 epoch 都会影响边际价值。比较模型时至少同时记录参数、激活参数、训练 token、精度、硬件和估算 FLOPs。
<!-- bilingual-en:start -->
Effective tokens are not raw bytes. Deduplication, quality, language and domain mixtures, contamination, and repeated epochs all affect marginal value. Model comparisons should jointly record total and active parameters, training tokens, numerical precision, hardware, and estimated FLOPs.
<!-- bilingual-en:end -->

“相同 token 数”也可能代表不同学习信号：高重复低质量语料的第二遍与新鲜高质量语料的第一遍价值不同。scaling 实验若同时改变数据过滤、tokenizer 或上下文长度，观察到的斜率就混合了多个因果因素。
<!-- bilingual-en:start -->
The same token count can represent different learning signals: a second pass over repetitive low-quality data differs from a first pass over fresh high-quality data. If a scaling experiment also changes filtering, tokenizer, or context length, its observed slope combines multiple causal factors.
<!-- bilingual-en:end -->

## Worked example：同一预算的两种分配
<!-- bilingual-en:start -->
*Worked Example: Two Allocations under One Budget*
<!-- bilingual-en:end -->

假设训练计算近似 $C=6ND$。方案 A 用 $N=10$ billion、$D=100$ billion，方案 B 用 $N=5$ billion、$D=200$ billion；二者 $ND$ 相同，粗略训练 FLOPs 相同。哪个 loss 更低不能仅由乘积判断，必须代入在相近数据和架构上拟合的 scaling 关系。
<!-- bilingual-en:start -->
Suppose training compute is approximated by $C=6ND$. Plan A uses $N=10$ billion parameters and $D=100$ billion tokens; Plan B uses $N=5$ billion and $D=200$ billion. Their products and rough training FLOPs match. The lower-loss choice cannot be inferred from the product alone; it requires a scaling relationship fitted on comparable data and architecture.
<!-- bilingual-en:end -->

若部署后预期生成一万亿 token，较小模型可能即使训练 loss 略高也拥有更低的总成本。训练计算最优、推理计算最优和成本—质量 Pareto 前沿是三个不同决策问题。
<!-- bilingual-en:start -->
If deployment is expected to generate one trillion tokens, the smaller model may yield lower total cost even with slightly higher training loss. Training-compute optimality, inference-compute optimality, and the cost–quality Pareto frontier are distinct decision problems.
<!-- bilingual-en:end -->

## 比较训练报告
<!-- bilingual-en:start -->
*Comparing Training Reports*
<!-- bilingual-en:end -->

先统一口径：dense 还是 MoE、总参数还是激活参数、预训练还是继续训练、tokenizer 与数据是否可比。没有数据组成、训练 token 和评测协议，单个参数数字只能说明很少。
<!-- bilingual-en:start -->
First align definitions: dense or MoE, total or active parameters, pretraining or continued training, and whether tokenizer and data are comparable. Without data composition, training tokens, and evaluation protocol, a lone parameter count says very little.
<!-- bilingual-en:end -->

## 失败诊断
<!-- bilingual-en:start -->
*Failure Diagnosis*
<!-- bilingual-en:end -->

- 先问拟合范围：外推是否超过已观测模型、token 或计算量一个数量级。
  <!-- bilingual-en:start -->
  Ask about the fitted range first: determine whether extrapolation extends an order of magnitude beyond observed models, tokens, or compute.
  <!-- bilingual-en:end -->
- 再问固定了什么：数据质量、优化器、上下文、架构和损失定义是否同时变化。
  <!-- bilingual-en:start -->
  Then ask what remained fixed: inspect whether data quality, optimizer, context, architecture, or loss definition changed simultaneously.
  <!-- bilingual-en:end -->
- 最后问决策目标：追求训练后最低 loss、给定延迟下最高质量，还是完整生命周期最低成本。
  <!-- bilingual-en:start -->
  Finally ask what decision is optimized: lowest post-training loss, highest quality under a latency constraint, or lowest lifecycle cost.
  <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum Self-Check*
<!-- bilingual-en:end -->

### 固定训练预算下，为什么参数越大未必越好？
<!-- bilingual-en:start -->
*Why are more parameters not always better under a fixed training budget?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 更大模型消耗更多每 token 计算，可能被迫看到更少数据而训练不足；需联合选择参数与 token。
<!-- bilingual-en:start -->
> [!answer]- Answer
> A larger model costs more per token and may therefore see too little data and remain undertrained; parameters and tokens must be chosen jointly.
<!-- bilingual-en:end -->

### 比较两个模型训练规模时至少要问哪些口径？
<!-- bilingual-en:start -->
*Which definitions must be checked when comparing two model-training scales?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 总/激活参数、训练 token、数据组成、数值精度、训练 FLOPs/硬件，以及 dense 或 MoE。
<!-- bilingual-en:start -->
> [!answer]- Answer
> Check total and active parameters, training tokens, data composition, numerical precision, training FLOPs and hardware, and whether the model is dense or MoE.
<!-- bilingual-en:end -->

### 某 scaling 曲线拟合很好，为什么仍不能叫自然定律？
<!-- bilingual-en:start -->
*Why is a well-fitted scaling curve still not a law of nature?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 它来自有限范围和特定数据、架构、优化配方；分布或配方变化、远距离外推和能力指标都可能破坏原关系。
<!-- bilingual-en:start -->
> [!answer]- Answer
> It comes from a finite range under a particular data distribution, architecture, and optimization recipe. Distribution or recipe changes, distant extrapolation, and capability metrics can break the relationship.
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
- Kaplan et al. (2020)、Hoffmann et al. (2022)、Fedus et al. (2022)：核验 scaling、compute-optimal 与 sparse MoE。
  <!-- bilingual-en:start -->
  Kaplan et al. (2020), Hoffmann et al. (2022), and Fedus et al. (2022) verify scaling, compute-optimal allocation, and sparse MoE respectively.
  <!-- bilingual-en:end -->
- [Kaplan et al. (2020), Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)：核验 loss 与模型、数据、计算规模的经验幂律及适用实验范围。
  <!-- bilingual-en:start -->
  [Kaplan et al. (2020), Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) verifies empirical power laws between loss and model, data, and compute scale within the studied regime.
  <!-- bilingual-en:end -->
- [Hoffmann et al. (2022), Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)：核验固定计算下参数与 token 的重新分配及 Chinchilla 实验。
  <!-- bilingual-en:start -->
  [Hoffmann et al. (2022), Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) verifies reallocation between parameters and tokens under fixed compute and the Chinchilla experiments.
  <!-- bilingual-en:end -->
