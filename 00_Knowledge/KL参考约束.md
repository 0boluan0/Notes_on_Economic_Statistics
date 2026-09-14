---
aliases:
  - "KL 参考约束是在策略优化目标中惩罚当前策略相对固定参考策略的条件分布偏离"
  - "A KL reference constraint penalizes divergence of the current policy's conditional distribution from a fixed reference policy in the optimization objective"
student_os: knowledge-atom
atom_id: LLM-PT-006
atom_set: llm-post-training
atom_type: definition
status: source-checked
mastery_state: unassessed
related:
  - "[[显式奖励模型RLHF]]"
  - "[[DPO目标]]"
leads_to:
  - "[[KL不保证奖励有效]]"
part_of:
  - "[[LLM 后训练.canvas|LLM 后训练]]"
---

# KL 参考约束是在策略优化目标中惩罚当前策略相对固定参考策略的条件分布偏离
<!-- bilingual-en:start -->
*A KL reference constraint penalizes divergence of the current policy's conditional distribution from a fixed reference policy in the optimization objective*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 给定提示 $x$、当前策略 $\pi_\theta(\cdot\mid x)$ 和固定参考策略 $\pi_{\mathrm{ref}}(\cdot\mid x)$，常见的奖励优化目标写成
> $$
> \max_\theta\;
> \mathbb E_{y\sim\pi_\theta(\cdot\mid x)}[r(x,y)]
> -\beta D_{\mathrm{KL}}\!\left(
> \pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)
> \right),\qquad \beta>0.
> $$
> 第二项就是 KL 参考约束：当前策略把概率移到参考策略很少生成的回答上时，通常要付出更高代价。它在“追求奖励”和“留在参考分布附近”之间建立交换。
>
> <!-- bilingual-en:start -->
> Given a prompt $x$, current policy $\pi_\theta$, and fixed reference policy $\pi_{\mathrm{ref}}$, reward optimization often maximizes expected reward minus $\beta D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})$ for $\beta>0$. The second term is the KL reference constraint: moving probability toward responses rare under the reference generally incurs a larger cost. It creates a trade-off between pursuing reward and remaining near the reference distribution.
> <!-- bilingual-en:end -->

## 约束读取概率比
<!-- bilingual-en:start -->
*The constraint reads probability ratios*
<!-- bilingual-en:end -->

对离散回答分布，正向 KL 为
$$
D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})
=\sum_y\pi_\theta(y\mid x)
\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}.
$$
它由当前策略下回答的概率加权，并比较每个回答在两种策略下的概率比。参考策略通常取 SFT 模型或另一个冻结策略；“参考”表示优化时的比较基准，不表示它天然正确。

<!-- bilingual-en:start -->
For a discrete response distribution, forward KL averages the log probability ratio under the current policy. The reference is often an SFT model or another frozen policy. “Reference” means the comparison baseline used by the objective, not a claim that the baseline is inherently correct.
<!-- bilingual-en:end -->

系数 $\beta$ 决定交换率。$\beta$ 增大时，同样的分布偏离更昂贵，策略通常更难离开参考；$\beta$ 减小时，奖励项相对更强。这个解释必须连同奖励尺度一起读：把奖励整体放大而保持 $\beta$ 不变，也会改变两项的相对力量。

<!-- bilingual-en:start -->
The coefficient $\beta$ sets the exchange rate. A larger value makes the same divergence more expensive and normally keeps the policy closer to the reference; a smaller value gives the reward term more relative influence. This must be read together with reward scale, because rescaling reward while holding $\beta$ fixed also changes the trade-off.
<!-- bilingual-en:end -->

## 名称相同不代表实现口径相同
<!-- bilingual-en:start -->
*The same name can hide different implementations*
<!-- bilingual-en:end -->

阅读实现时至少要核对三件事：KL 的方向；是对整段回答分布的理论量，还是由采样 token 对数比估计；系数是固定还是自适应。精确序列级 KL、逐 token 样本估计和额外 loss 项可以服务相同设计目的，却不是逐值相等的统计量。

DPO 的标准推导也从 KL 正则奖励目标出发，但训练时把奖励—策略关系重参数化进成对损失，并不运行显式 PPO 管线。因此“含有参考策略”不能用来判断一个系统一定采用哪种优化算法。

<!-- bilingual-en:start -->
An implementation must state the KL direction, whether it uses a sequence-level quantity or sampled token log ratios, and whether the coefficient is fixed or adaptive. These variants can serve the same design role without being numerically identical. Standard DPO is derived from a KL-regularized reward objective but reparameterizes the reward-policy relation into a pairwise loss, so the presence of a reference policy does not identify one optimization algorithm.
<!-- bilingual-en:end -->

> [!example] 最小例子
> 参考策略给回答 A、B 的概率是 $(0.5,0.5)$，当前策略改成 $(0.9,0.1)$。当前策略把更多质量集中到 A，因而相对参考产生正的 KL 代价；奖励若更偏爱 A，优化器就在奖励收益与这项偏离代价之间权衡。
>
> <!-- bilingual-en:start -->
> A reference assigns probabilities $(0.5,0.5)$ to responses A and B, while the current policy assigns $(0.9,0.1)$. Concentrating more mass on A creates a positive KL cost relative to the reference. If reward favors A, the optimizer trades that gain against the divergence penalty.
> <!-- bilingual-en:end -->

> [!question]- 自检
> 保持奖励和参考策略不变，只把 $\beta$ 调大。目标函数直接增加了哪一类压力？能否据此断言回答会更真实？
>
> <!-- bilingual-en:start -->
> Hold reward and the reference policy fixed and increase $\beta$. What pressure changes directly? Does this establish that responses become more truthful?
> <!-- bilingual-en:end -->
>
> **答案：** 偏离参考策略的代价增大，当前策略通常会被压得更接近参考分布。真实性不是 KL 公式读取的变量；这种约束能保证到哪里，见 [[KL不保证奖励有效]]。
>
> <!-- bilingual-en:start -->
> **Answer:** Divergence from the reference becomes more expensive, normally keeping the current policy closer to it. Truthfulness is not read by the KL formula; [[KL不保证奖励有效|the guarantee boundary]] treats what this constraint cannot establish.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Ouyang et al. (2022), *Training language models to follow instructions with human feedback*](https://arxiv.org/html/2203.02155#S3.SS4)：第 3.4 节与附录 C 给出相对 SFT 参考策略的 KL 惩罚及系数设置。
- [Rafailov et al. (2023), *Direct Preference Optimization*](https://arxiv.org/html/2305.18290#S3)：第 3 节从 KL 正则奖励最大化目标推导闭式最优策略和 DPO 参数化。

<!-- bilingual-en:start -->
- Ouyang et al. (2022), §3.4 and Appendix C, documents a KL penalty relative to an SFT reference policy and its coefficient.
- Rafailov et al. (2023), §3, derives the closed-form optimizer and DPO parameterization from KL-regularized reward maximization.
<!-- bilingual-en:end -->
