---
aliases:
  - "逐 token 全局损失均值与逐样本损失均值会给不同长度序列不同权重，因此不能互换"
  - A global per-token loss mean and a mean of per-sample loss means weight unequal-length sequences differently and are not interchangeable
student_os: knowledge-atom
atom_id: LLM-PRE-016
atom_set: llm-pretraining
atom_type: measurement-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[自回归目标]]"
related:
  - "[[Token口径]]"
  - "[[SFT目标]]"
  - "[[SFT损失掩码]]"
leads_to:
  - "[[困惑度]]"
part_of:
  - "[[LLM 预训练.canvas|LLM 预训练]]"
---

# 逐 token 全局损失均值与逐样本损失均值会给不同长度序列不同权重，因此不能互换

<!-- bilingual-en:start -->
*A global per-token loss mean and a mean of per-sample loss means weight unequal-length sequences differently and are not interchangeable*
<!-- bilingual-en:end -->

> [!summary] 聚合规则是目标的一部分
> 令第 $i$ 条序列的有效位置集合为 $V_i$，逐位置 NLL 为 $\ell_{it}$，并先排除 $|V_i|=0$ 的样本。记 $I_+=\{i:|V_i|>0\}$、$B_+=|I_+|$，并要求 $B_+>0$。全局 token 均值是
> $$
> L_{\mathrm{token}}=\frac{\sum_{i\in I_+}\sum_{t\in V_i}\ell_{it}}{\sum_{i\in I_+}|V_i|},
> $$
> 而逐样本均值再平均是
> $$
> L_{\mathrm{sample}}=\frac1{B_+}\sum_{i\in I_+}\frac{\sum_{t\in V_i}\ell_{it}}{|V_i|}.
> $$
> 当各样本有效长度不同，两式给出的权重和梯度一般不同。
>
> <!-- bilingual-en:start -->
> After excluding samples with no valid targets, a global token mean sums all valid-position NLL terms and divides once by the valid-token count. A mean of per-sample means first normalizes each remaining sequence and then gives each sequence equal weight. Unequal valid lengths make these different objectives.
> <!-- bilingual-en:end -->

## 一个最小反例

样本 A 只有 1 个有效 token，平均 NLL 为 0；样本 B 有 9 个有效 token，每个 NLL 都是 1。全局 token 均值为 $9/10=0.9$，逐样本均值为 $(0+1)/2=0.5$。两者都可以被明确选择，但不能报告成同一个量。

同理，直接平均不同 batch 或不同设备已经归一化的 loss，会让有效 token 较少的 batch/设备获得额外权重。若目标是全局 token micro-average，应累计分子和分母，再只除一次。自然对数给出 nats/token，以 2 为底的对数给出 bits/token。

<!-- bilingual-en:start -->
If one sample has one valid token with mean NLL 0 and another has nine valid tokens with mean NLL 1, the global token mean is $0.9$ while the mean of sample means is $0.5$. Averaging already-normalized batch or device losses creates the same weighting issue. A global token micro-average must aggregate numerator and denominator before division.
<!-- bilingual-en:end -->

> [!question]- 自检
> 两个 batch 的平均 NLL 都是 1 和 3，但有效 token 数分别是 100 和 10。全局 token 均值能否写成 $(1+3)/2$？
>
> **答案：** 不能。全局 token 均值是 $(100\times1+10\times3)/110\approx1.18$；简单平均 2 给了小 batch 过高权重。

## 来源与核验

- [PyTorch `CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)：核对 `mean` reduction 对非 ignored 目标项归一化，以及 `ignore_index` 不进入分子和分母。
- [[自回归目标]]：提供逐位置 next-token NLL；本卡只拥有跨位置、样本与 batch 的聚合边界。
