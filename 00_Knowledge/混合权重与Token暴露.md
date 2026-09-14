---
aliases:
  - "来源权重只有结合抽样单位与加载规则，才能换算为期望 token 暴露"
  - "预训练数据混合的抽样单位与来源权重共同决定期望 token 暴露，而不是由原始语料大小自动决定"
  - "Source weights can be converted to expected token exposure only together with the sampling unit and loading rules"
student_os: knowledge-atom
atom_id: LLM-PRE-020
atom_set: llm-pretraining
atom_type: measurement-boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[LLM 预训练.canvas|LLM 预训练]]"
requires:
  - "[[数据混合]]"
  - "[[Token口径]]"
leads_to:
  - "[[预训练分布]]"
---

# 来源权重只有结合抽样单位与加载规则，才能换算为期望 token 暴露
<!-- bilingual-en:start -->
*Source weights can be converted to expected token exposure only together with the sampling unit and loading rules*
<!-- bilingual-en:end -->

> [!summary] 口径边界
> 若 sampler 每次先以概率 $w_i$ 选择来源 $i$，再输出等长 token block，并计划训练 $T$ 个 token，则
> $$
> \mathbb E[T_i]=w_iT.
> $$
> 这个等式依赖“权重作用于等长 token block”。若权重作用于不等长文档或 example，$w_i$ 就不再直接等于 token 占比。
>
> <!-- bilingual-en:start -->
> If a sampler selects source $i$ with probability $w_i$, emits an equal-length token block, and plans to train on $T$ tokens, then $\mathbb E[T_i]=w_iT$. The equality depends on weights acting on equal-length token blocks. With variable-length documents or examples, $w_i$ need not equal token share.
> <!-- bilingual-en:end -->

## 为什么等长条件重要
<!-- bilingual-en:start -->
*Why equal length matters*
<!-- bilingual-en:end -->

若一共抽取 $N$ 个长度为 $L$ 的 block，$T=NL$。令 $I_k$ 表示第 $k$ 个 block 的来源，且每次来源选择满足 $\Pr(I_k=i)=w_i$，则

$$
T_i=L\sum_{k=1}^{N}\mathbf 1\{I_k=i\},
\qquad
\mathbb E[T_i]=L\sum_{k=1}^{N}w_i=w_iT.
$$

若按不等长文档独立抽样，来源 $i$ 的平均文档长度为 $\mu_i$，在无截断、无耗尽等理想条件下，长期 token 占比更接近

$$
\frac{w_i\mu_i}{\sum_jw_j\mu_j},
$$

而不是 $w_i$。真实实现还会受 packing、无放回抽样、来源耗尽和最后一个 block 截断影响，因此最终应核验实际 token 计数。
<!-- bilingual-en:start -->
For $N$ blocks of length $L$, $T=NL$ and $T_i=L\sum_k\mathbf 1\{I_k=i\}$, so expectation gives $w_iT$ when each source choice has marginal probability $w_i$. With independently sampled variable-length documents of mean length $\mu_i$, long-run token share is instead closer to $w_i\mu_i/\sum_jw_j\mu_j$ under idealized no-truncation and no-exhaustion conditions. Packing, sampling without replacement, source exhaustion, and final-block truncation require checking realized token counts.
<!-- bilingual-en:end -->

## 等效遍历次数的额外前提
<!-- bilingual-en:start -->
*Extra assumptions for equivalent passes*
<!-- bilingual-en:end -->

令 $U_i$ 为来源 $i$ 在同一清洗、去重和 tokenizer 口径下可供抽样的 **token 实例总数**，不是词表里不同 token 类型的数量。当 $w_i$ 已表示 token 暴露份额，而且加载规则能够实现计划中的 $w_iT$ 个 token 暴露时，计划口径的期望等效遍历次数为

$$
e_i^{\mathrm{plan}}=\frac{w_iT}{U_i}.
$$

不放回抽样并不自动使这个比值失效：若 $w_iT\le U_i$ 且来源尚未耗尽，它只表示计划抽取的库存比例，可能小于 1。若来源先耗尽而 loader 又不循环或有放回地继续抽样，计划暴露便无法实现；此时必须用实际观察到的 $T_i^{\mathrm{real}}$ 报告

$$
e_i^{\mathrm{real}}=\frac{T_i^{\mathrm{real}}}{U_i},
$$

不能继续套用 $w_iT$。无论哪种口径，等效遍历次数都是暴露量与可抽样 token 实例总数的比值，不表示每个实例恰好都出现相同次数；TB 大小也不能直接代替 $U_i$。
<!-- bilingual-en:start -->
Let $U_i$ be the number of sampleable token instances under matched cleaning, deduplication, and tokenization, not the number of distinct vocabulary types. If $w_i$ is a token-exposure share and the loader can realize the planned exposure $w_iT$, planned expected equivalent passes are $e_i^{\mathrm{plan}}=w_iT/U_i$. Sampling without replacement does not itself invalidate the ratio while $w_iT\le U_i$ and the source remains unexhausted; the value then represents the planned fraction of inventory sampled. If the source exhausts and the loader neither cycles nor samples with replacement, the plan cannot be realized, so report $e_i^{\mathrm{real}}=T_i^{\mathrm{real}}/U_i$ from observed exposure instead. Neither ratio means every instance is seen the same number of times, and byte size cannot replace $U_i$.
<!-- bilingual-en:end -->

> [!example] 最小例子
> A、B 经相同处理后各有 10 亿个可抽样 token 实例；训练使用等长 block、允许循环抽样，总预算 100 亿 token，权重为 80%/20%。期望暴露为 80 亿/20 亿 token，等效遍历约 8/2 次。少了任一前提，都不能直接照搬这个结果。
>
> <!-- bilingual-en:start -->
> A and B each have one billion sampleable token instances. Training uses equal-length blocks, allows cycling, has a ten-billion-token budget, and assigns 80%/20% weights. Expected exposure is eight and two billion tokens, or about eight and two equivalent passes. Remove any stated assumption and the calculation may no longer apply.
> <!-- bilingual-en:end -->

> [!question]- 自检
> GPT-3 表 2.2 的 mixture weight 被定义为抽到某数据集的 example 比例。能否在不知道 example 长度时直接把它叫作 token 比例？
>
> **答案：** 不能。必须先确认 example 是否等长，或查看论文另行报告的 token 与 epoch 口径。

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Brown et al. (2020), *Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165)：表 2.2 把抽到某数据集的 example 比例、token 数和训练遍历次数分列，直接要求先核对权重单位。
- [Grattafiori et al. (2024), *The Llama 3 Herd of Models*](https://arxiv.org/abs/2407.21783)：第 3.1.2 节以 token 类别比例报告最终 mixture，说明来源份额只有绑定计数口径后才有确定含义。
